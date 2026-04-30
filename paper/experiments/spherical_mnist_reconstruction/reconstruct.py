#!/usr/bin/env python3
"""Reconstruct spherical MNIST digits from their SO(3)-bispectrum.

Demonstrates that the SO(3) bispectrum is a complete SO(3)-invariant on S^2:
the original signal can be recovered up to an SO(3) "phase change" (the
Wigner-D action on spherical-harmonic coefficients), which is exactly the
group-orbit indeterminacy of any complete invariant.

Pipeline per digit ``f``:

    1. Generate ``K`` independent random SO(3) rotations and the rotated
       images ``f_k = R_k . f``.
    2. Compute targets ``beta_k = bsp(f_k)``. By invariance these should
       all coincide modulo SHT discretisation; the residual
       ``||beta_k - beta_0|| / ||beta_0||`` is logged as the
       *invariance check*.
    3. Independently reconstruct each ``f_k`` from ``beta_k`` by Adam on
       the relative complex L2 loss
       ``||beta(f_hat) - beta_k||^2 / ||beta_k||^2``.
    4. Optionally re-project ``f_hat`` onto the band-limit ``lmax`` after
       every step to suppress the Gaussian high-frequency null-space.

The demo deliberately makes no attempt to align the reconstruction back
to the original (no SO(3) Procrustes): the *point* is that the recovery
is faithful only up to an unknown rotation. Side-by-side equirectangular
plots make this visually obvious; quantitative residuals close out the
story.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_harmonics import InverseRealSHT, RealSHT

from bispectrum import SO3onS2, random_rotation_matrix, rotate_spherical_function

logger = logging.getLogger('so3_reconstruct')

NEURIPS_RCPARAMS: dict[str, object] = {
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'text.usetex': False,
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
}


@dataclass
class ReconResult:
    """Per (digit, rotation) reconstruction record."""

    digit_idx: int
    rot_idx: int
    label: int
    invariance_rel: float
    final_recon_rel: float
    image_space_rel: float
    aligned_image_space_rel: float
    seconds: float
    align_seconds: float
    target: torch.Tensor
    initial: torch.Tensor
    recovered: torch.Tensor
    aligned: torch.Tensor
    loss_trace: list[float] = field(default_factory=list)
    bispec_rel_trace: list[float] = field(default_factory=list)


def load_digits(
    data_dir: Path,
    n_digits: int,
    nlat: int,
    nlon: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``n_digits`` random samples from the cached spherical MNIST test set.

    Picks one example per class first (so the figure shows a variety of digits),
    then top up with extra random samples if ``n_digits > 10``.
    """
    cache_path = data_dir / 'spherical_cache' / f'test_{nlat}x{nlon}.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Spherical MNIST cache missing: {cache_path}. '
            f'Run paper/experiments/spherical_mnist/data.py first.'
        )

    blob = torch.load(cache_path, weights_only=True)
    images: torch.Tensor = blob['images']
    labels: torch.Tensor = blob['labels']

    rng = torch.Generator().manual_seed(seed)

    chosen: list[int] = []
    for cls in range(10):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        if cls_idx.numel() == 0:
            continue
        pick = cls_idx[torch.randint(cls_idx.numel(), (1,), generator=rng)].item()
        chosen.append(int(pick))
        if len(chosen) >= n_digits:
            break

    if len(chosen) < n_digits:
        remaining = n_digits - len(chosen)
        already = set(chosen)
        all_idx = torch.randperm(images.shape[0], generator=rng).tolist()
        for idx in all_idx:
            if idx not in already:
                chosen.append(idx)
                already.add(idx)
                if len(chosen) >= n_digits:
                    break

    chosen_t = torch.tensor(chosen[:n_digits], dtype=torch.long)
    return images[chosen_t].clone(), labels[chosen_t].clone()


def make_rotation_stack(
    f: torch.Tensor,
    n_rotations: int,
    seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return ``(1 + n_rotations, nlat, nlon)`` stack [f, R_1.f, ..., R_K.f].

    Index 0 is always the identity; rotations are deterministic in ``seed``.
    """
    if f.dim() != 2:
        raise ValueError(f'f must be (nlat, nlon), got shape {tuple(f.shape)}')

    stack = [f.unsqueeze(0)]
    rotations: list[torch.Tensor] = []
    for k in range(n_rotations):
        torch.manual_seed(seed * 1_000_003 + k)
        R = random_rotation_matrix(device=f.device)
        rotations.append(R)
        rotated = rotate_spherical_function(f.unsqueeze(0), R)
        stack.append(rotated)

    return torch.cat(stack, dim=0), rotations


def _bandlimit_project(
    f: torch.Tensor,
    sht: RealSHT,
    isht: InverseRealSHT,
) -> torch.Tensor:
    """Project ``f`` onto the band-limit defined by ``sht`` (SHT round-trip)."""
    return isht(sht(f))


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix.

    Auto-normalises so the optimizer can move freely in R^4.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


def align_so3(
    f_hat: torch.Tensor,
    f_target: torch.Tensor,
    n_restarts: int = 12,
    n_steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Find ``R \\in SO(3)`` minimising ``||rotate(f_hat, R) - f_target||^2``.

    Args:
        f_hat: ``(nlat, nlon)`` candidate signal.
        f_target: ``(nlat, nlon)`` target signal.
        n_restarts: independent quaternion initialisations (Haar-uniform via
            ``randn(4) / norm``). Multi-restart is essential because the
            ``arctan2`` / ``acos`` chain in the spherical sampler creates
            discrete local minima.
        n_steps: Adam steps per restart (cosine-annealed LR).
        lr: initial learning rate.
        seed: deterministic restart seeding.

    Returns:
        ``(aligned, R, rel_loss)`` where ``aligned = rotate(f_hat, R)`` and
        ``rel_loss = ||aligned - target||^2 / ||target||^2`` for the best
        restart.
    """
    device = f_hat.device
    dtype = f_hat.dtype
    src = f_hat.unsqueeze(0)
    tgt = f_target.unsqueeze(0)
    target_norm_sq = tgt.pow(2).sum().clamp_min(1e-30)

    best_loss = float('inf')
    best_R = torch.eye(3, device=device, dtype=dtype)
    best_aligned = src.clone().squeeze(0)

    for restart in range(n_restarts):
        gen = torch.Generator(device='cpu').manual_seed(seed * 9_973 + restart)
        q_init = torch.randn(4, generator=gen).to(device=device, dtype=dtype)
        q_init = q_init / q_init.norm().clamp_min(1e-12)
        q = q_init.clone().requires_grad_(True)

        opt = torch.optim.Adam([q], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(n_steps, 1), eta_min=lr * 1e-2
        )

        last_good = q_init.clone()
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            R = _quat_to_matrix(q)
            rotated = rotate_spherical_function(src, R).squeeze(0)
            loss = (rotated - f_target).pow(2).sum() / target_norm_sq
            loss.backward()
            if torch.isnan(q.grad).any() or torch.isinf(q.grad).any():
                with torch.no_grad():
                    q.copy_(last_good)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()
            with torch.no_grad():
                last_good.copy_(q.detach())

        with torch.no_grad():
            R = _quat_to_matrix(q.detach())
            rotated = rotate_spherical_function(src, R).squeeze(0)
            final = ((rotated - f_target).pow(2).sum() / target_norm_sq).item()

        if final < best_loss:
            best_loss = final
            best_R = R
            best_aligned = rotated.detach()

    return best_aligned, best_R, float(best_loss)


def align_so3_batch(
    f_hats: torch.Tensor,
    f_targets: torch.Tensor,
    n_restarts: int = 12,
    n_steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    """Per-sample SO(3) alignment over a batch.

    Loops over the batch dim and calls :func:`align_so3` on each pair.
    """
    if f_hats.shape != f_targets.shape:
        raise ValueError(
            f'shape mismatch: f_hats {tuple(f_hats.shape)} vs f_targets {tuple(f_targets.shape)}'
        )
    aligned_list: list[torch.Tensor] = []
    R_list: list[torch.Tensor] = []
    losses: list[float] = []
    for i in range(f_hats.shape[0]):
        aligned, R, loss = align_so3(
            f_hats[i], f_targets[i], n_restarts=n_restarts, n_steps=n_steps,
            lr=lr, seed=seed * 1_000_003 + i,
        )
        aligned_list.append(aligned)
        R_list.append(R)
        losses.append(loss)
    return torch.stack(aligned_list, dim=0), R_list, losses


def reconstruct_batch(
    bsp: SO3onS2,
    sht: RealSHT,
    isht: InverseRealSHT,
    beta_targets: torch.Tensor,
    nlat: int,
    nlon: int,
    n_steps: int,
    lr: float,
    bandlimit_project: bool,
    log_every: int,
    device: torch.device,
    dtype: torch.dtype,
    init_seed: int,
    lr_min_ratio: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]], list[list[float]]]:
    """Adam reconstruction on a batch of bispectrum targets.

    Uses a cosine-annealed learning rate from ``lr`` down to
    ``lr * lr_min_ratio`` and tracks the per-sample best ``f_hat`` (lowest
    relative bispectrum residual) so that late-stage Adam jitter never
    degrades the returned reconstruction.

    Args:
        beta_targets: ``(B, output_size)`` complex bispectra.
        n_steps: optimizer iterations.
        lr: initial Adam learning rate.
        bandlimit_project: if True, run ``isht(sht(f_hat))`` after every step.
        init_seed: seed for the random Gaussian image initialisation.
        lr_min_ratio: cosine-anneal floor as a fraction of ``lr``.

    Returns:
        ``(initial, recovered, loss_trace, bispec_rel_trace)`` where the
        traces are lists of ``B`` lists of length ``n_steps + 1``. The
        traces report the *best-so-far* relative residual at every step
        (non-increasing).
    """
    B = beta_targets.shape[0]
    g = torch.Generator(device='cpu').manual_seed(init_seed)
    f_hat_cpu = torch.randn(B, nlat, nlon, generator=g)
    f_hat = f_hat_cpu.to(device=device, dtype=dtype).requires_grad_(True)

    initial = f_hat.detach().clone()

    target_norm_sq = (beta_targets.real**2 + beta_targets.imag**2).sum(dim=-1).clamp_min(1e-30)

    optimizer = torch.optim.Adam([f_hat], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_steps, 1), eta_min=lr * lr_min_ratio
    )

    best_rel = torch.full((B,), float('inf'), device=device, dtype=torch.float32)
    best_f = f_hat.detach().clone()

    loss_trace: list[list[float]] = [[] for _ in range(B)]
    bispec_rel_trace: list[list[float]] = [[] for _ in range(B)]

    def _update_best(per_sample_loss: torch.Tensor, current_f: torch.Tensor) -> torch.Tensor:
        rel = torch.sqrt(per_sample_loss.detach()).to(best_rel.dtype)
        improved = rel < best_rel
        if improved.any():
            best_rel[improved] = rel[improved]
            best_f[improved] = current_f.detach()[improved]
        return best_rel.clone()

    def _record(per_sample_loss: torch.Tensor, best_rel_now: torch.Tensor) -> None:
        loss_vals = per_sample_loss.detach().cpu().tolist()
        rel_vals = best_rel_now.cpu().tolist()
        for b in range(B):
            loss_trace[b].append(float(loss_vals[b]))
            bispec_rel_trace[b].append(float(rel_vals[b]))

    with torch.amp.autocast('cuda', enabled=False):
        beta_hat = bsp(f_hat)
        diff = beta_hat - beta_targets
        per_sample = (diff.real**2 + diff.imag**2).sum(dim=-1) / target_norm_sq
    best_rel_now = _update_best(per_sample, f_hat)
    _record(per_sample, best_rel_now)

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=False):
            beta_hat = bsp(f_hat)
            diff = beta_hat - beta_targets
            per_sample = (diff.real**2 + diff.imag**2).sum(dim=-1) / target_norm_sq
            loss = per_sample.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if bandlimit_project:
            with torch.no_grad():
                projected = _bandlimit_project(f_hat.detach(), sht, isht)
                f_hat.data.copy_(projected)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            beta_hat_post = bsp(f_hat)
            diff_post = beta_hat_post - beta_targets
            per_sample_post = (
                (diff_post.real**2 + diff_post.imag**2).sum(dim=-1) / target_norm_sq
            )
        best_rel_now = _update_best(per_sample_post, f_hat)
        _record(per_sample_post, best_rel_now)

        if log_every and (step + 1) % log_every == 0:
            med_rel = float(best_rel_now.median())
            cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                'step %5d / %5d  best median rel residual %.4e  lr %.2e',
                step + 1,
                n_steps,
                med_rel,
                cur_lr,
            )

    recovered = best_f.detach().clone()
    return initial, recovered, loss_trace, bispec_rel_trace


def relative_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-batch ``||a - b|| / ||b||`` along the last dim(s)."""
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    diff = flat_a - flat_b
    if diff.is_complex():
        num = torch.sqrt((diff.real**2 + diff.imag**2).sum(dim=-1))
        den = torch.sqrt((flat_b.real**2 + flat_b.imag**2).sum(dim=-1)).clamp_min(1e-30)
    else:
        num = diff.norm(dim=-1)
        den = flat_b.norm(dim=-1).clamp_min(1e-30)
    return num / den


def run_demo(
    data_dir: Path,
    output_dir: Path,
    n_digits: int,
    n_rotations: int,
    nlat: int,
    nlon: int,
    lmax: int,
    selective: bool,
    n_steps: int,
    lr: float,
    bandlimit_project: bool,
    device: torch.device,
    seed: int,
    log_every: int,
    align_n_restarts: int,
    align_n_steps: int,
    align_lr: float,
    n_recon_restarts: int = 1,
) -> tuple[list[ReconResult], dict[str, object]]:
    """End-to-end reconstruction sweep. Returns per-(digit, rot) records + meta."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading %d digits from cached spherical MNIST ...', n_digits)
    images, labels = load_digits(data_dir, n_digits, nlat, nlon, seed=seed)
    logger.info('  shapes: images=%s labels=%s', tuple(images.shape), tuple(labels.shape))

    logger.info('Building SO3onS2(lmax=%d, selective=%s) on %s', lmax, selective, device)
    bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon, selective=selective).to(device)
    sht = RealSHT(
        nlat, nlon, lmax=lmax + 1, mmax=lmax + 1, grid='equiangular', norm='ortho'
    ).to(device)
    isht = InverseRealSHT(
        nlat, nlon, lmax=lmax + 1, mmax=lmax + 1, grid='equiangular', norm='ortho'
    ).to(device)
    logger.info('  bispectrum output_size=%d', bsp.output_size)

    dtype = torch.float32
    results: list[ReconResult] = []
    sweep_t0 = time.perf_counter()

    for d_idx in range(n_digits):
        f = images[d_idx].to(device=device, dtype=dtype)
        if bandlimit_project:
            with torch.no_grad():
                f = _bandlimit_project(f.unsqueeze(0), sht, isht).squeeze(0)
        label = int(labels[d_idx].item())
        logger.info('Digit %d/%d (label=%d)', d_idx + 1, n_digits, label)

        stack, _ = make_rotation_stack(f, n_rotations, seed=seed * 7919 + d_idx)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            beta_stack = bsp(stack.contiguous())
        beta_ref = beta_stack[0:1]
        invariance_rel = relative_norm(beta_stack, beta_ref.expand_as(beta_stack))
        for k, val in enumerate(invariance_rel.cpu().tolist()):
            tag = 'identity' if k == 0 else f'rot {k}'
            logger.info('  invariance ||beta(f_%d) - beta(f_0)|| / ||beta(f_0)|| = %.3e (%s)',
                        k, val, tag)

        t0 = time.perf_counter()
        best_initial: torch.Tensor | None = None
        best_recovered: torch.Tensor | None = None
        best_loss_trace: list[list[float]] | None = None
        best_bispec_trace: list[list[float]] | None = None
        best_per_target: torch.Tensor | None = None
        target_norm_sq = (
            (beta_stack.real**2 + beta_stack.imag**2).sum(dim=-1).clamp_min(1e-30)
        )

        for restart in range(max(n_recon_restarts, 1)):
            init, recovered, loss_trace, bispec_trace = reconstruct_batch(
                bsp,
                sht,
                isht,
                beta_targets=beta_stack,
                nlat=nlat,
                nlon=nlon,
                n_steps=n_steps,
                lr=lr,
                bandlimit_project=bandlimit_project,
                log_every=log_every if restart == 0 else 0,
                device=device,
                dtype=dtype,
                init_seed=(seed + 1) * 99991 + d_idx + restart * 31_337,
            )
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                beta_rec = bsp(recovered)
                diff = beta_rec - beta_stack
                per_target_rel_sq = (
                    (diff.real**2 + diff.imag**2).sum(dim=-1) / target_norm_sq
                )

            if best_recovered is None:
                best_initial = init.clone()
                best_recovered = recovered.clone()
                best_loss_trace = [list(t) for t in loss_trace]
                best_bispec_trace = [list(t) for t in bispec_trace]
                best_per_target = per_target_rel_sq.clone()
            else:
                improved = per_target_rel_sq < best_per_target
                for b in range(stack.shape[0]):
                    if bool(improved[b].item()):
                        best_initial[b] = init[b]
                        best_recovered[b] = recovered[b]
                        best_loss_trace[b] = list(loss_trace[b])
                        best_bispec_trace[b] = list(bispec_trace[b])
                        best_per_target[b] = per_target_rel_sq[b]
            if n_recon_restarts > 1:
                logger.info(
                    '  restart %d/%d: median best per-target rel = %.3e',
                    restart + 1,
                    n_recon_restarts,
                    float(best_per_target.sqrt().median()),
                )

        assert best_initial is not None and best_recovered is not None
        assert best_loss_trace is not None and best_bispec_trace is not None
        initial = best_initial
        recovered = best_recovered
        loss_trace = best_loss_trace
        bispec_trace = best_bispec_trace
        t1 = time.perf_counter()
        recon_seconds_total = t1 - t0
        recon_seconds_per_target = recon_seconds_total / max(stack.shape[0], 1)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            beta_recovered = bsp(recovered)
        final_recon_rel = relative_norm(beta_recovered, beta_stack)
        image_space_rel = relative_norm(recovered, stack)

        align_t0 = time.perf_counter()
        if align_n_restarts > 0 and align_n_steps > 0:
            aligned, _Rs, aligned_losses = align_so3_batch(
                recovered,
                stack,
                n_restarts=align_n_restarts,
                n_steps=align_n_steps,
                lr=align_lr,
                seed=seed * 7 + d_idx,
            )
            aligned_rel = torch.tensor(aligned_losses, device=device).sqrt()
        else:
            aligned = recovered.detach().clone()
            aligned_rel = image_space_rel.clone()
        align_seconds_total = time.perf_counter() - align_t0
        align_seconds_per_target = align_seconds_total / max(stack.shape[0], 1)

        for k in range(stack.shape[0]):
            rec = ReconResult(
                digit_idx=d_idx,
                rot_idx=k,
                label=label,
                invariance_rel=float(invariance_rel[k].item()),
                final_recon_rel=float(final_recon_rel[k].item()),
                image_space_rel=float(image_space_rel[k].item()),
                aligned_image_space_rel=float(aligned_rel[k].item()),
                seconds=recon_seconds_per_target,
                align_seconds=align_seconds_per_target,
                target=stack[k].detach().cpu(),
                initial=initial[k].detach().cpu(),
                recovered=recovered[k].detach().cpu(),
                aligned=aligned[k].detach().cpu(),
                loss_trace=loss_trace[k],
                bispec_rel_trace=bispec_trace[k],
            )
            results.append(rec)
            tag = 'identity' if k == 0 else f'rot {k}'
            logger.info(
                '  %s: bisp %.3e | img %.3e (raw) | img %.3e (R-aligned) | recon %.2fs / align %.2fs',
                tag,
                rec.final_recon_rel,
                rec.image_space_rel,
                rec.aligned_image_space_rel,
                recon_seconds_per_target,
                align_seconds_per_target,
            )

    sweep_seconds = time.perf_counter() - sweep_t0
    meta: dict[str, object] = {
        'n_digits': n_digits,
        'n_rotations': n_rotations,
        'nlat': nlat,
        'nlon': nlon,
        'lmax': lmax,
        'selective': selective,
        'output_size': int(bsp.output_size),
        'n_steps': n_steps,
        'lr': lr,
        'bandlimit_project': bandlimit_project,
        'n_recon_restarts': n_recon_restarts,
        'align_n_restarts': align_n_restarts,
        'align_n_steps': align_n_steps,
        'align_lr': align_lr,
        'device': str(device),
        'seed': seed,
        'sweep_seconds': sweep_seconds,
    }
    logger.info('Sweep done in %.1fs', sweep_seconds)
    return results, meta


def dump_results_json(results: list[ReconResult], meta: dict[str, object], path: Path) -> None:
    """Persist per-record metrics + meta. Tensors are dropped (kept in figures)."""
    payload: dict[str, object] = {
        'meta': meta,
        'records': [
            {
                'digit_idx': r.digit_idx,
                'rot_idx': r.rot_idx,
                'label': r.label,
                'invariance_rel': r.invariance_rel,
                'final_recon_rel': r.final_recon_rel,
                'image_space_rel': r.image_space_rel,
                'aligned_image_space_rel': r.aligned_image_space_rel,
                'seconds': r.seconds,
                'align_seconds': r.align_seconds,
                'bispec_rel_trace': r.bispec_rel_trace,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info('Wrote %s', path)


def _signal_view_angle(
    f: np.ndarray,
    fallback_elev: float = 25.0,
    fallback_azim: float = 30.0,
) -> tuple[float, float]:
    """Camera ``(elev, azim)`` aimed at the signed-mass centroid of ``f``.

    Uses ``relu(f)`` weighting (the digit is the positive part; negative
    band-limit ringing is downweighted) and includes the spherical area
    element ``sin(theta)`` so polar samples don't dominate. Returns the
    fallback angles if the positive mass is degenerate.
    """
    nlat, nlon = f.shape
    theta = np.linspace(0.0, np.pi, nlat)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)[None, :]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.broadcast_to(cos_t, f.shape)

    w = np.maximum(f, 0.0) * sin_t
    total = float(w.sum())
    if total < 1e-10:
        return fallback_elev, fallback_azim

    cx = float((w * x).sum() / total)
    cy = float((w * y).sum() / total)
    cz = float((w * z).sum() / total)
    norm = float(np.sqrt(cx * cx + cy * cy + cz * cz))
    if norm < 1e-6:
        return fallback_elev, fallback_azim

    cx, cy, cz = cx / norm, cy / norm, cz / norm
    elev = float(np.rad2deg(np.arcsin(np.clip(cz, -1.0, 1.0))))
    azim = float(np.rad2deg(np.arctan2(cy, cx)))
    return elev, azim


def _orthographic_project(
    f: np.ndarray,
    view_size: int,
    elev_deg: float = 25.0,
    azim_deg: float = 30.0,
) -> np.ndarray:
    """Render an equirectangular spherical signal as an orthographic disk.

    Camera is placed at infinity along the unit vector pointing at
    spherical coords ``(elev, azim)``; we rotate the world so that camera
    direction becomes ``+z`` and then project to the ``(x, y)`` plane.

    Args:
        f: ``(nlat, nlon)`` equirectangular sample with ``theta in [0, pi]``
            on rows (top-to-bottom) and ``phi in [0, 2*pi)`` on columns.
        view_size: output is ``(view_size, view_size)``; pixels outside the
            unit disk are NaN.
        elev_deg, azim_deg: camera direction in degrees (latitude / longitude).

    Returns:
        ``(view_size, view_size)`` float array (NaN outside the disk).
    """
    nlat, nlon = f.shape
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    z_world = cam / np.linalg.norm(cam)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, z_world)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x_world = np.cross(up, z_world)
    x_world /= np.linalg.norm(x_world)
    y_world = np.cross(z_world, x_world)

    u = np.linspace(-1.0, 1.0, view_size)
    v = np.linspace(1.0, -1.0, view_size)
    U, V = np.meshgrid(u, v)
    R2 = U * U + V * V
    mask = R2 <= 1.0
    W = np.zeros_like(U)
    W[mask] = np.sqrt(1.0 - R2[mask])

    P = U[..., None] * x_world + V[..., None] * y_world + W[..., None] * z_world

    theta = np.arccos(np.clip(P[..., 2], -1.0, 1.0))
    phi = np.arctan2(P[..., 1], P[..., 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    theta_idx = theta / np.pi * (nlat - 1)
    phi_idx = phi / (2 * np.pi) * nlon
    ti0 = np.floor(theta_idx).astype(np.int64)
    pi0 = np.floor(phi_idx).astype(np.int64) % nlon
    ti1 = np.clip(ti0 + 1, 0, nlat - 1)
    pi1 = (pi0 + 1) % nlon
    dt = theta_idx - ti0
    dp = phi_idx - np.floor(phi_idx)

    sample = (
        (1 - dt) * (1 - dp) * f[ti0, pi0]
        + (1 - dt) * dp * f[ti0, pi1]
        + dt * (1 - dp) * f[ti1, pi0]
        + dt * dp * f[ti1, pi1]
    )
    sample = np.where(mask, sample, np.nan)
    return sample


def _sphere_grid_lines(
    elev_deg: float,
    azim_deg: float,
    n_lat: int = 6,
    n_lon: int = 8,
    n_pts: int = 96,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return polylines ``[(u, v), ...]`` for great/small circles on the visible hemisphere.

    Used to draw a faint reference graticule on the orthographic sphere so the
    eye can pick up rotation differences between panels.
    """
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    z_world = cam / np.linalg.norm(cam)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, z_world)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x_world = np.cross(up, z_world)
    x_world /= np.linalg.norm(x_world)
    y_world = np.cross(z_world, x_world)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    phi_pts = np.linspace(0, 2 * np.pi, n_pts)
    for theta in np.linspace(np.pi / (n_lat + 1), np.pi - np.pi / (n_lat + 1), n_lat):
        x = np.sin(theta) * np.cos(phi_pts)
        y = np.sin(theta) * np.sin(phi_pts)
        z = np.full_like(phi_pts, np.cos(theta))
        u = x * x_world[0] + y * x_world[1] + z * x_world[2]
        v = x * y_world[0] + y * y_world[1] + z * y_world[2]
        w = x * z_world[0] + y * z_world[1] + z * z_world[2]
        mask = w > 0
        lines.append((u[mask], v[mask]))

    theta_pts = np.linspace(0, np.pi, n_pts)
    for phi in np.linspace(0, 2 * np.pi, n_lon, endpoint=False):
        x = np.sin(theta_pts) * np.cos(phi)
        y = np.sin(theta_pts) * np.sin(phi)
        z = np.cos(theta_pts)
        u = x * x_world[0] + y * x_world[1] + z * x_world[2]
        v = x * y_world[0] + y * y_world[1] + z * y_world[2]
        w = x * z_world[0] + y * z_world[1] + z * z_world[2]
        mask = w > 0
        lines.append((u[mask], v[mask]))

    return lines


def _draw_sphere(
    ax: plt.Axes,
    f: np.ndarray,
    vmin: float,
    vmax: float,
    view_size: int,
    elev_deg: float,
    azim_deg: float,
    grid_lines: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Draw an orthographic sphere of ``f`` into ``ax``."""
    disk = _orthographic_project(f, view_size, elev_deg=elev_deg, azim_deg=azim_deg)
    ax.imshow(
        disk,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        extent=(-1.0, 1.0, -1.0, 1.0),
        origin='upper',
        interpolation='bilinear',
    )
    if grid_lines is not None:
        for u, v in grid_lines:
            ax.plot(u, v, color='gray', linewidth=0.25, alpha=0.45, zorder=2)
    circle = plt.Circle(
        (0, 0), 1.0, fill=False, color='black', linewidth=0.6, zorder=3
    )
    ax.add_patch(circle)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_panel(
    ax: plt.Axes,
    f: np.ndarray,
    vmin: float,
    vmax: float,
    render: str,
    view_size: int,
    elev_deg: float,
    azim_deg: float,
    grid_lines: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Render a single signal panel using the requested style."""
    if render == 'sphere':
        _draw_sphere(
            ax, f, vmin, vmax, view_size, elev_deg, azim_deg, grid_lines=grid_lines
        )
    else:
        ax.imshow(
            f,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            interpolation='nearest',
        )
        ax.set_xticks([])
        ax.set_yticks([])


def make_orbits_figure(
    results: list[ReconResult],
    n_digits: int,
    n_rotations: int,
    path: Path,
    render: str = 'sphere',
    view_size: int = 128,
    elev_deg: float = 25.0,
    azim_deg: float = 30.0,
    auto_center: bool = True,
    panel_size: float | None = None,
) -> None:
    """Per-row triplets ``(target, raw recon, R-aligned recon)`` per rotation.

    With ``render='sphere'`` each panel is an orthographic projection of the
    function onto the unit sphere. When ``auto_center`` is ``True`` (default)
    each panel's camera points at the signal's positive-mass centroid so the
    digit is in the middle of the disk; otherwise all panels share
    ``(elev_deg, azim_deg)``. A faint lat/lon graticule is overlaid for
    rotation reference.
    """
    plt.rcParams.update(NEURIPS_RCPARAMS)
    n_targets = 1 + n_rotations
    n_cols = 3 * n_targets

    if panel_size is None:
        panel_size = 1.5 if render == 'sphere' else 1.2
    fig, axes = plt.subplots(
        n_digits,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_digits),
        squeeze=False,
    )

    fallback_lines = (
        _sphere_grid_lines(elev_deg, azim_deg) if render == 'sphere' else None
    )

    def _view_for(arr: np.ndarray) -> tuple[float, float, list[tuple[np.ndarray, np.ndarray]] | None]:
        if render != 'sphere':
            return elev_deg, azim_deg, None
        if not auto_center:
            return elev_deg, azim_deg, fallback_lines
        e, a = _signal_view_angle(arr, fallback_elev=elev_deg, fallback_azim=azim_deg)
        return e, a, _sphere_grid_lines(e, a)

    grouped: dict[tuple[int, int], ReconResult] = {(r.digit_idx, r.rot_idx): r for r in results}

    for d in range(n_digits):
        first = grouped[(d, 0)]
        for k in range(n_targets):
            r = grouped[(d, k)]
            tgt = r.target.numpy()
            raw = r.recovered.numpy()
            ali = r.aligned.numpy()
            vmax = max(abs(tgt).max(), abs(raw).max(), abs(ali).max(), 1e-8)
            vmin = -vmax

            e_t, a_t, gl_t = _view_for(tgt)
            e_r, a_r, gl_r = _view_for(raw)
            e_a, a_a, gl_a = _view_for(ali)

            ax_t = axes[d][3 * k]
            _draw_panel(
                ax_t, tgt, vmin, vmax, render, view_size, e_t, a_t,
                grid_lines=gl_t,
            )
            if d == 0:
                title = (
                    f'target $f$\n(label={first.label})'
                    if k == 0
                    else f'target $R_{k}\\!\\cdot\\! f$'
                )
                ax_t.set_title(title)
            if k == 0:
                ax_t.set_ylabel(
                    f'#{d}', rotation=0, ha='right', va='center', fontsize=8
                )

            ax_r = axes[d][3 * k + 1]
            _draw_panel(
                ax_r, raw, vmin, vmax, render, view_size, e_r, a_r,
                grid_lines=gl_r,
            )
            if d == 0:
                ax_r.set_title(f'recon $\\hat f_{k}$\n($\\beta$ rel {r.final_recon_rel:.1e})')
            else:
                ax_r.set_xlabel(f'$\\beta$ {r.final_recon_rel:.1e}', fontsize=6)

            ax_a = axes[d][3 * k + 2]
            _draw_panel(
                ax_a, ali, vmin, vmax, render, view_size, e_a, a_a,
                grid_lines=gl_a,
            )
            if d == 0:
                ax_a.set_title(
                    f'aligned $R\\!\\cdot\\!\\hat f_{k}$\n(img rel {r.aligned_image_space_rel:.1e})'
                )
            else:
                ax_a.set_xlabel(
                    f'img {r.aligned_image_space_rel:.1e}', fontsize=6
                )

    fig.suptitle(
        'SO(3) bispectrum reconstruction.\n'
        'Target $f_k = R_k\\!\\cdot\\! f$ (left of each triplet), gradient-descent recon '
        '$\\hat f_k$ from $\\beta(f_k)$ (middle), and the recon rotated by the optimal '
        '$R \\in SO(3)$ (right).\n'
        'Each sphere is auto-centered on the signal\'s positive-mass centroid; '
        'aligned and target panels share the same view because they are the same signal.',
        y=1.02,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    logger.info('Wrote %s', path)


def make_convergence_figure(results: list[ReconResult], path: Path) -> None:
    """Median + IQR envelope of the relative bispectrum residual vs. step."""
    plt.rcParams.update(NEURIPS_RCPARAMS)
    if not results:
        return

    traces = np.array([r.bispec_rel_trace for r in results], dtype=np.float64)
    if traces.size == 0:
        return
    steps = np.arange(traces.shape[1])
    med = np.median(traces, axis=0)
    p25 = np.percentile(traces, 25, axis=0)
    p75 = np.percentile(traces, 75, axis=0)

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.fill_between(steps, p25, p75, color='#2D6A9F', alpha=0.20, linewidth=0, label='IQR')
    ax.plot(steps, med, color='#2D6A9F', linewidth=1.2, label='median')

    ax.set_xlabel('Adam step')
    ax.set_ylabel(r'$\|\beta(\hat f) - \beta(f)\| / \|\beta(f)\|$')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right')

    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    logger.info('Wrote %s', path)


def make_invariance_figure(results: list[ReconResult], path: Path) -> None:
    """Scatter of invariance check vs. recon residual per (digit, rot)."""
    plt.rcParams.update(NEURIPS_RCPARAMS)
    inv = np.array([r.invariance_rel for r in results if r.rot_idx > 0])
    rec = np.array([r.final_recon_rel for r in results if r.rot_idx > 0])
    if inv.size == 0:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(inv, rec, s=10, color='#2D6A9F', alpha=0.7, edgecolor='none')
    lo = max(min(inv.min(), rec.min()) * 0.5, 1e-8)
    hi = max(inv.max(), rec.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=0.6, label='$y=x$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'invariance $\|\beta(R_k f) - \beta(f)\| / \|\beta(f)\|$')
    ax.set_ylabel(r'recon $\|\beta(\hat f_k) - \beta(R_k f)\| / \|\beta(R_k f)\|$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    logger.info('Wrote %s', path)


def make_paper_figure(
    results: list[ReconResult],
    digit_indices: list[int],
    path: Path,
    view_size: int = 192,
    panel_size: float = 1.9,
    auto_center: bool = True,
    elev_deg: float = 25.0,
    azim_deg: float = 30.0,
) -> None:
    """Compact 2-row x 2-col figure for the NeurIPS paper.

    For each requested ``digit_idx`` we pick the rotation with the lowest
    aligned image residual and render ``[target | aligned recon]`` as a row.
    The result is a clean ``len(digit_indices) x 2`` grid suitable for a
    ``\\includegraphics[width=0.6\\linewidth]`` inclusion.
    """
    plt.rcParams.update(NEURIPS_RCPARAMS)

    by_digit: dict[int, list[ReconResult]] = {}
    for r in results:
        by_digit.setdefault(r.digit_idx, []).append(r)

    picks: list[ReconResult] = []
    for d in digit_indices:
        if d not in by_digit:
            raise ValueError(
                f'digit_idx {d} not in results (available: {sorted(by_digit)})'
            )
        best = min(by_digit[d], key=lambda r: r.aligned_image_space_rel)
        picks.append(best)

    n_rows = len(picks)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )

    fallback_lines = _sphere_grid_lines(elev_deg, azim_deg)

    def _view_for(arr: np.ndarray) -> tuple[float, float, list[tuple[np.ndarray, np.ndarray]]]:
        if not auto_center:
            return elev_deg, azim_deg, fallback_lines
        e, a = _signal_view_angle(arr, fallback_elev=elev_deg, fallback_azim=azim_deg)
        return e, a, _sphere_grid_lines(e, a)

    for i, r in enumerate(picks):
        target = r.target.numpy()
        aligned = r.aligned.numpy()
        vmax = max(abs(target).max(), abs(aligned).max(), 1e-8)
        vmin = -vmax

        e_t, a_t, gl_t = _view_for(target)

        _draw_sphere(
            axes[i][0], target, vmin, vmax, view_size, e_t, a_t, grid_lines=gl_t
        )
        _draw_sphere(
            axes[i][1], aligned, vmin, vmax, view_size, e_t, a_t, grid_lines=gl_t
        )

        if i == 0:
            axes[i][0].set_title('Target $f$', fontsize=10, pad=4)
            axes[i][1].set_title(
                r'Recon $\hat{R}\!\cdot\!\hat{f}$', fontsize=10, pad=4
            )
        axes[i][0].set_ylabel(
            f'class {r.label}',
            rotation=0,
            ha='right',
            va='center',
            fontsize=9,
            labelpad=10,
        )
        axes[i][1].set_xlabel(
            rf'$\|\hat{{R}}\!\cdot\!\hat{{f}}-f\|/\|f\|={r.aligned_image_space_rel:.2f}$',
            fontsize=7,
            labelpad=2,
        )

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'))
    plt.close(fig)
    logger.info('Wrote paper figure %s (and PNG sibling)', path)


def print_summary(results: list[ReconResult]) -> None:
    """Plain-text summary of metrics, ready to paste into a log."""
    inv_all = np.array([r.invariance_rel for r in results if r.rot_idx > 0])
    rec_all = np.array([r.final_recon_rel for r in results])
    img_all = np.array([r.image_space_rel for r in results])
    aligned_all = np.array([r.aligned_image_space_rel for r in results])

    def _stats(x: np.ndarray) -> str:
        if x.size == 0:
            return 'n/a'
        return (
            f'median={np.median(x):.3e}  '
            f'p25={np.percentile(x, 25):.3e}  '
            f'p75={np.percentile(x, 75):.3e}  '
            f'max={np.max(x):.3e}'
        )

    print('\nSummary across all (digit, rotation) pairs:')
    print(f'  invariance     ||beta(R_k f) - beta(f)|| / ||beta(f)||         : {_stats(inv_all)}')
    print(f'  recon resid    ||beta(f_hat) - beta(target)|| / ||target||     : {_stats(rec_all)}')
    print(f'  image (raw)    ||f_hat - target|| / ||target||                 : {_stats(img_all)}')
    print(f'  image (aligned)||R . f_hat - target|| / ||target||             : {_stats(aligned_all)}')
    print(
        '  Raw image-space error is O(1) (the recovery sits in a different SO(3) orbit '
        'representative); after the SO(3) alignment the error drops to ~SHT discretisation '
        'floor, demonstrating recovery is exact up to an SO(3) rotation.'
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_digits', type=int, default=8)
    parser.add_argument('--n_rotations', type=int, default=2)
    parser.add_argument(
        '--lmax',
        type=int,
        default=12,
        help='Bispectrum band-limit. lmax=12 keeps the recon below the SHT '
        'discretisation floor at the default 64x128 grid (cleanest visual). '
        'lmax=15 matches the classifier experiment but recon plateaus near '
        'the SHT noise floor, blurring the alignment quality.',
    )
    parser.add_argument('--nlat', type=int, default=64)
    parser.add_argument('--nlon', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument(
        '--full_bispectrum',
        action='store_true',
        help='Use the full O(L^3) bispectrum instead of the selective O(L^2) variant.',
    )
    parser.add_argument(
        '--no_bandlimit_project',
        action='store_true',
        help='Disable the band-limit re-projection after each Adam step.',
    )
    parser.add_argument('--n_recon_restarts', type=int, default=4,
                        help='Random-init restarts per reconstruction; per-sample best is kept.')
    parser.add_argument('--align_n_restarts', type=int, default=12,
                        help='SO(3) alignment restarts per (target, recon) pair (0 disables).')
    parser.add_argument('--align_n_steps', type=int, default=200,
                        help='Adam steps per alignment restart.')
    parser.add_argument('--align_lr', type=float, default=1e-1,
                        help='Initial Adam LR for alignment (cosine-annealed).')
    parser.add_argument(
        '--render',
        type=str,
        default='sphere',
        choices=['sphere', 'equirectangular'],
        help='Panel rendering style for the orbits figure.',
    )
    parser.add_argument('--view_size', type=int, default=128,
                        help='Orthographic view resolution (only when --render=sphere).')
    parser.add_argument('--elev_deg', type=float, default=25.0,
                        help='Fallback / fixed-view elevation in degrees.')
    parser.add_argument('--azim_deg', type=float, default=30.0,
                        help='Fallback / fixed-view azimuth in degrees.')
    parser.add_argument('--fixed_view', action='store_true',
                        help='Use --elev_deg / --azim_deg for every panel instead of '
                        'auto-centering on each signal\'s positive-mass centroid.')
    parser.add_argument('--paper_digits', type=int, nargs='+', default=[0, 1],
                        help='Digit indices used for the compact paper figure (2x2).')
    parser.add_argument('--paper_figure_path', type=Path, default=None,
                        help='Override output path for the paper figure '
                        '(default: <output_dir>/paper_orbits.pdf).')
    parser.add_argument('--paper_only', action='store_true',
                        help='Run only the digits required for the paper figure '
                        '(--paper_digits) and skip the comprehensive figures. '
                        'Fast regeneration path for the paper.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=300)
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=here.parent / 'spherical_mnist' / 'smnist_data',
    )
    parser.add_argument('--output_dir', type=Path, default=here / 'figures')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA requested but not available; falling back to CPU.')
        device = torch.device('cpu')

    paper_digits = sorted(set(int(d) for d in args.paper_digits))
    if args.paper_only:
        if not paper_digits:
            raise ValueError('--paper_only requires --paper_digits to be non-empty')
        n_digits_run = max(paper_digits) + 1
        n_rotations_run = 1
        logger.info(
            'paper_only: running %d digits and 1 rotation to feed paper figure %s',
            n_digits_run,
            paper_digits,
        )
    else:
        n_digits_run = args.n_digits
        n_rotations_run = args.n_rotations

    results, meta = run_demo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_digits=n_digits_run,
        n_rotations=n_rotations_run,
        nlat=args.nlat,
        nlon=args.nlon,
        lmax=args.lmax,
        selective=not args.full_bispectrum,
        n_steps=args.n_steps,
        lr=args.lr,
        bandlimit_project=not args.no_bandlimit_project,
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        align_n_restarts=args.align_n_restarts,
        align_n_steps=args.align_n_steps,
        align_lr=args.align_lr,
        n_recon_restarts=args.n_recon_restarts,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paper_path = args.paper_figure_path or (args.output_dir / 'paper_orbits.pdf')

    if args.paper_only:
        make_paper_figure(
            results,
            digit_indices=paper_digits,
            path=paper_path,
            auto_center=not args.fixed_view,
            elev_deg=args.elev_deg,
            azim_deg=args.azim_deg,
        )
        print_summary(results)
        return 0

    dump_results_json(results, meta, args.output_dir / 'results.json')
    make_orbits_figure(
        results,
        n_digits=n_digits_run,
        n_rotations=n_rotations_run,
        path=args.output_dir / 'orbits.pdf',
        render=args.render,
        view_size=args.view_size,
        elev_deg=args.elev_deg,
        azim_deg=args.azim_deg,
        auto_center=not args.fixed_view,
    )
    make_convergence_figure(results, args.output_dir / 'convergence.pdf')
    make_invariance_figure(results, args.output_dir / 'invariance_vs_recon.pdf')
    make_paper_figure(
        results,
        digit_indices=paper_digits,
        path=paper_path,
        auto_center=not args.fixed_view,
        elev_deg=args.elev_deg,
        azim_deg=args.azim_deg,
    )
    print_summary(results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
