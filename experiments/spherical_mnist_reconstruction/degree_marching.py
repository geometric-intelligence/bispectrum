#!/usr/bin/env python3
"""Recover spherical signals from the augmented selective SO(3) invariant, one degree at a time.

Post hoc companion to ``reconstruct.py``. Gradient descent on the image from noise stalls in
spurious stationary points; this solver works on real spherical-harmonic coefficients instead,
in float64 on the CPU:

    1. Levenberg-Marquardt (LM) on the degrees ``<= seed_degree`` against the invariant entries
       that involve only those degrees, from random starts, until ``n_seed_branches`` solutions
       are found. Generically each is a rotation of the true low-degree part.
    2. For ``ell = seed_degree + 1 .. lmax``: LM on degree ``ell`` alone against the entries whose
       largest degree is ``ell``, lower degrees frozen. With ``joint_polish``, every accepted
       solution is then refined by LM on all degrees ``<= ell`` against all entries up to ``ell``.
    3. LM on everything, then each surviving branch is scored: selective and full-bispectrum
       residuals, and the image residual after SO(3) alignment to the target.

A branch that matches the invariant to ``ok_tol`` but does not align onto the target would be a
collision, two SO(3) orbits with one invariant; every surviving branch is recorded.

Usage:
    python degree_marching.py solve --signal_source smnist --n_signals 8 --n_rotations 1 \\
        --output_dir OUT --targets 0 1 2
    python degree_marching.py merge --output_dir OUT
    python degree_marching.py plot --results OUT/results.json --paper_figure_path FIG.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from reconstruct import (  # noqa: F401  (ReconResult unpickles reconstruct.py state files)
    ReconResult,
    _bandlimit_project,
    load_digits,
    make_orbits_figure,
    make_paper_figure,
    make_random_bandlimited_signals,
    make_rotation_stack,
)
from torch_harmonics import InverseRealSHT, RealSHT

from bispectrum import SO3onS2
from bispectrum.rotation import (
    create_spherical_grid,
    rotate_spherical_function,
    spherical_to_cartesian,
)

logger = logging.getLogger('degree_marching')

DT = torch.float64
CDT = torch.complex128


# --------------------------------------------------------------------------
# Real spherical-harmonic coordinates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealSHBasis:
    """Real coordinates ``x`` of a band-limited real signal, isometric to ``L2(S^2)``.

    Per degree ``l``: ``Re f_l0``, then ``Re f_lm, Im f_lm`` for ``m = 1..l`` (scaled by sqrt 2).
    """

    lmax: int
    l_of_dof: torch.Tensor
    m_of_dof: torch.Tensor
    imag_dof: torch.Tensor

    @classmethod
    def build(cls, lmax: int) -> RealSHBasis:
        ls: list[int] = []
        ms: list[int] = []
        imag: list[bool] = []
        for ell in range(lmax + 1):
            ls.append(ell)
            ms.append(0)
            imag.append(False)
            for m in range(1, ell + 1):
                ls += [ell, ell]
                ms += [m, m]
                imag += [False, True]
        return cls(lmax, torch.tensor(ls), torch.tensor(ms), torch.tensor(imag))

    @property
    def dim(self) -> int:
        return (self.lmax + 1) ** 2

    def to_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, dim)`` real -> ``(B, lmax + 1, lmax + 1)`` complex ortho half-spectrum."""
        batch = x.shape[0]
        n = self.lmax + 1
        xs = x * torch.where(self.m_of_dof == 0, 1.0, 2.0**-0.5).to(x.dtype)
        rows = torch.arange(batch)[:, None]
        real = ~self.imag_dof
        re = torch.zeros(batch, n, n, dtype=x.dtype).index_put(
            (rows, self.l_of_dof[real][None], self.m_of_dof[real][None]), xs[:, real]
        )
        im = torch.zeros(batch, n, n, dtype=x.dtype).index_put(
            (rows, self.l_of_dof[self.imag_dof][None], self.m_of_dof[self.imag_dof][None]),
            xs[:, self.imag_dof],
        )
        return torch.complex(re, im)

    def from_coeffs(self, c: torch.Tensor) -> torch.Tensor:
        picked = c[:, self.l_of_dof, self.m_of_dof]
        vals = torch.where(self.imag_dof, picked.imag, picked.real)
        return vals * torch.where(self.m_of_dof == 0, 1.0, 2.0**0.5).to(vals.dtype)

    def degree_power(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.lmax + 1, dtype=x.dtype).index_add_(0, self.l_of_dof, x**2)


# --------------------------------------------------------------------------
# Invariant, exact Jacobian, LM
# --------------------------------------------------------------------------


class InvariantModel:
    """The selective invariant as a map ``x -> R^{2N}`` (real and imaginary parts), with its Jacobian.

    Reads the private sparse tables of :class:`SO3onS2`; :meth:`check_jacobian` guards against
    them drifting out of sync with this code.
    """

    def __init__(self, lmax: int, nlat: int, nlon: int) -> None:
        self.basis = RealSHBasis.build(lmax)
        self.nlat, self.nlon = nlat, nlon
        self.bsp = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon, selective=True).to(DT)
        self.full = SO3onS2(lmax=lmax, nlat=nlat, nlon=nlon, selective=False).to(DT)
        self.sht = RealSHT(
            nlat, nlon, lmax=lmax + 1, mmax=lmax + 1, grid='equiangular', norm='ortho'
        ).to(DT)
        self.isht = InverseRealSHT(
            nlat, nlon, lmax=lmax + 1, mmax=lmax + 1, grid='equiangular', norm='ortho'
        ).to(DT)
        self.n_entries = self.bsp.output_size
        degree = torch.tensor([max(t) for t in self.bsp.index_map])
        self.row_degree = torch.cat([degree, degree])
        self._build_jacobian_tables()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bsp._forward_sparse_fast(
            self.basis.to_coeffs(x).to(CDT), x.shape[0], self.n_entries
        )
        return torch.cat([z.real, z.imag], dim=-1)

    def grid(self, x: torch.Tensor) -> torch.Tensor:
        return self.isht(self.basis.to_coeffs(x))

    def coefficients(self, g: torch.Tensor) -> torch.Tensor:
        return self.basis.from_coeffs(self.sht(g))

    def project(self, g: torch.Tensor) -> torch.Tensor:
        return self.isht(self.sht(g))

    def full_bispectrum_rel(self, x: torch.Tensor, x_ref: torch.Tensor) -> float:
        a, b = self.full(self.grid(x[None])), self.full(self.grid(x_ref[None]))
        return float((a - b).norm() / b.norm())

    def _build_jacobian_tables(self) -> None:
        dim = self.basis.dim
        self.features(torch.zeros(1, dim, dtype=DT))
        self.flat_map = self.bsp._sht_to_flat(
            self.basis.to_coeffs(torch.eye(dim, dtype=DT)).to(CDT)
        )
        n_flat = self.flat_map.shape[1]
        self.flat_dof = torch.zeros(n_flat, 2, dtype=torch.long)
        self.flat_w = torch.zeros(n_flat, 2, dtype=CDT)
        nonzero = self.flat_map.abs() > 0
        for j in range(n_flat):
            dofs = torch.nonzero(nonzero[:, j]).flatten().tolist()
            for s, d in enumerate(dofs):
                self.flat_dof[j, s] = d
                self.flat_w[j, s] = self.flat_map[d, j]
            if len(dofs) == 1:
                self.flat_dof[j, 1] = dofs[0]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Exact real Jacobian ``(2N, dim)`` of :meth:`features` at a single ``x``."""
        b = self.bsp
        n, dim = self.n_entries, self.basis.dim
        flat = x.to(CDT) @ self.flat_map
        jc = torch.zeros(n, dim, dtype=CDT)
        i1, i2, i3, eid, cg = b._sc_bi_fl1, b._sc_bi_fl2, b._sc_bi_fl, b._sc_bi_eid, b._sc_bi_cg
        for idx, coef, conj in (
            (i1, cg * flat[i2] * flat[i3].conj(), False),
            (i2, cg * flat[i1] * flat[i3].conj(), False),
            (i3, cg * flat[i1] * flat[i2], True),
        ):
            for s in range(2):
                w = self.flat_w[idx, s]
                jc.index_put_(
                    (eid, self.flat_dof[idx, s]), coef * (w.conj() if conj else w), accumulate=True
                )
        jr = torch.cat([jc.real, jc.imag], 0)
        if b._sc_pw_n_rows > 0:
            p1, p2, pcg, cidx = b._sc_pw_fl1, b._sc_pw_fl2, b._sc_pw_cg, b._sc_pw_cidx
            n_pw, mc = b._sc_pw_n_rows, b._sc_pw_max_coupled
            coupled = torch.zeros(n_pw * mc, dtype=CDT)
            coupled.index_add_(0, cidx, flat[p1] * flat[p2] * pcg)
            cbar = coupled.conj()[cidx]
            row_eid = b._sc_pw_eids[cidx // mc]
            jp = torch.zeros(n, dim, dtype=DT)
            for idx, other in ((p1, p2), (p2, p1)):
                base = cbar * pcg * flat[other]
                for s in range(2):
                    jp.index_put_(
                        (row_eid, self.flat_dof[idx, s]),
                        2.0 * (base * self.flat_w[idx, s]).real,
                        accumulate=True,
                    )
            power_rows = torch.zeros(n, dtype=torch.bool)
            power_rows[b._sc_pw_eids] = True
            jr[:n][power_rows] = jp[power_rows]
            jr[n:][power_rows] = 0.0
        return jr

    def check_jacobian(self, seed: int = 0, tol: float = 1e-10) -> float:
        """Relative difference between :meth:`jacobian` and forward-mode autodiff at a random point."""
        from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

        dim = self.basis.dim
        x = torch.randn(dim, generator=torch.Generator().manual_seed(seed), dtype=DT)
        with dual_level():
            xd = make_dual(x[None].expand(dim, -1).contiguous(), torch.eye(dim, dtype=DT))
            _, tangent = unpack_dual(self.features(xd))
        reference = tangent.T
        rel = float((self.jacobian(x) - reference).norm() / reference.norm())
        if rel > tol:
            raise RuntimeError(
                f'analytic Jacobian disagrees with autodiff: rel {rel:.2e} > {tol:.0e}'
            )
        return rel


@dataclass
class LMResult:
    x: torch.Tensor
    rel: float
    iters: int


def lm_solve(
    model: InvariantModel,
    x0: torch.Tensor,
    target: torch.Tensor,
    *,
    dof_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    iters: int = 400,
    lam0: float = 1e-2,
    tol: float = 1e-13,
) -> LMResult:
    """LM on ``||F(x) - target|| / ||target||`` over ``dof_mask``, restricted to ``row_mask``."""
    if dof_mask is None:
        dof_mask = torch.ones(model.basis.dim, dtype=torch.bool)
    if row_mask is None:
        row_mask = torch.ones(target.shape[0], dtype=torch.bool)
    target = target[row_mask]
    tnorm = float(target.norm())
    x = x0.clone()

    def resid(xx: torch.Tensor) -> torch.Tensor:
        return model.features(xx[None])[0][row_mask] - target

    r = resid(x)
    cost = float(r @ r)
    lam = lam0
    hist = [math.sqrt(cost) / tnorm]
    for _ in range(iters):
        jac = model.jacobian(x)[row_mask][:, dof_mask]
        g = jac.T @ r
        h = jac.T @ jac
        diag = torch.diagonal(h).clamp_min(1e-12 * float(torch.diagonal(h).max()))
        accepted = False
        while lam < 1e14:
            step = torch.linalg.solve(h + lam * torch.diag(diag), -g)
            xn = x.clone()
            xn[dof_mask] += step
            rn = resid(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 5.0, 1e-15)
                accepted = True
                break
            lam *= 4.0
        hist.append(math.sqrt(cost) / tnorm)
        if not accepted or hist[-1] < tol:
            break
        if len(hist) > 40 and hist[-1] > 0.999 * hist[-40]:
            break
    return LMResult(x=x, rel=hist[-1], iters=len(hist) - 1)


# --------------------------------------------------------------------------
# SO(3) alignment: closed form from quadrupole moments, then local refinement
# --------------------------------------------------------------------------


class MomentAligner:
    """Find ``R`` with ``P(R . h) ~ t`` (``P`` = band-limit projection) from quadrupole eigenframes.

    The four sign choices of the eigenframe give candidate rotations; the best two are refined
    by Adam on a quaternion. Unlike random-restart alignment this does not miss the basin.
    """

    def __init__(self, model: InvariantModel, refine_steps: int = 150) -> None:
        theta, phi = create_spherical_grid(model.nlat, model.nlon)
        self.xyz = spherical_to_cartesian(theta, phi).to(DT)
        self.weight = torch.sin(theta).to(DT)
        self.model = model
        self.refine_steps = refine_steps

    def _quadrupole(self, g: torch.Tensor) -> torch.Tensor:
        wg = (self.weight * g)[..., None, None]
        outer = self.xyz[..., :, None] * self.xyz[..., None, :]
        return (wg * (3 * outer - torch.eye(3, dtype=DT))).sum(dim=(0, 1))

    def _residual(self, h: torch.Tensor, t: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        moved = self.model.project(rotate_spherical_function(h[None], rot))[0]
        return ((moved - t) ** 2).sum() / (t**2).sum()

    def align(self, h: torch.Tensor, t: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Return ``(relative residual, R)`` for ``(nlat, nlon)`` float64 grids ``h`` and ``t``."""
        h, t = h.detach().contiguous(), t.detach().contiguous()
        _, vh = torch.linalg.eigh(self._quadrupole(h))
        _, vt = torch.linalg.eigh(self._quadrupole(t))
        candidates = []
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                rot = vt @ torch.diag(torch.tensor([s1, s2, 1.0], dtype=DT)) @ vh.T
                if torch.det(rot) < 0:
                    rot = vt @ torch.diag(torch.tensor([s1, s2, -1.0], dtype=DT)) @ vh.T
                candidates.append(rot)
        with torch.no_grad():
            scored = sorted(
                ((float(self._residual(h, t, r)), r) for r in candidates), key=lambda p: p[0]
            )
        best_loss, best_rot = scored[0]
        for _, rot0 in scored[:2]:
            q = _quat_from_matrix(rot0).requires_grad_(True)
            opt = torch.optim.Adam([q], lr=3e-3)
            for _ in range(self.refine_steps):
                opt.zero_grad()
                self._residual(h, t, _quat_to_matrix(q)).backward()
                opt.step()
            with torch.no_grad():
                rot = _quat_to_matrix(q)
                val = float(self._residual(h, t, rot))
            if val < best_loss:
                best_loss, best_rot = val, rot.detach()
        return best_loss**0.5, best_rot

    def apply(self, h: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        return self.model.project(rotate_spherical_function(h[None], rot))[0]


def _quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    t = m.trace()
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        q = torch.stack(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    else:
        i = int(torch.argmax(torch.diagonal(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = torch.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = torch.zeros(4, dtype=m.dtype)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    return (q / q.norm()).detach()


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm()
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


# --------------------------------------------------------------------------
# Degree-by-degree marching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarchConfig:
    seed_degree: int = 4
    n_seed_inits: int = 200
    n_seed_branches: int = 2
    n_extend_inits: int = 40
    max_solutions_per_branch: int = 2
    # From this degree on, stop at the first consistent solution per branch (None: never).
    single_solution_from_degree: int | None = None
    beam_width: int = 12
    ok_tol: float = 1e-10
    joint_polish: bool = True
    # A frozen per-degree solve below this goes on to the joint polish, which must reach ok_tol.
    frozen_accept_tol: float = 1e-6
    # When every branch dies at a degree, retry it with this many times more inits.
    retry_factor: int = 4
    lm_iters: int = 200
    final_lm_iters: int = 400


LEGACY_CONFIG = MarchConfig(
    single_solution_from_degree=10, joint_polish=False, frozen_accept_tol=1e-10, retry_factor=0
)


@dataclass
class DegreeLog:
    degree: int
    alive: int
    best_frozen_rel: float
    best_polished_rel: float | None
    retried: bool = False


@dataclass
class BranchScore:
    feature_rel: float
    full_bispectrum_rel: float
    aligned_image_rel: float


@dataclass
class TargetResult:
    target: int
    digit_idx: int
    rot_idx: int
    label: int
    seed_inits_tried: int
    n_seed_branches: int
    degrees: list[DegreeLog]
    branches: list[BranchScore]
    feature_rel: float | None
    full_bispectrum_rel: float | None
    aligned_image_rel: float | None
    recovered: bool
    collisions: int
    seconds: float
    x_target: list[float]
    x_recovered: list[float] | None
    rotation: list[list[float]] | None


def _seed_scale(
    model: InvariantModel, u: torch.Tensor, target: torch.Tensor, rows: torch.Tensor
) -> float:
    """Scale ``s`` with ``||F(s u)|| = ||target||`` on ``rows``; ``F`` is homogeneous and increasing in ``s``."""
    goal = float(target[rows].norm())
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(model.features((math.exp(mid) * u)[None])[0][rows].norm()) < goal:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def _distinct(
    cand: torch.Tensor, others: list[torch.Tensor], mask: torch.Tensor, scale: float
) -> bool:
    return all(float((cand[mask] - o[mask]).norm()) > 1e-6 * scale for o in others)


def seed_branches(
    model: InvariantModel, target: torch.Tensor, cfg: MarchConfig, gen: torch.Generator
) -> tuple[list[torch.Tensor], int]:
    """Solutions on degrees ``<= seed_degree`` from random starts; also returns the number of starts used."""
    basis = model.basis
    dofs = basis.l_of_dof <= cfg.seed_degree
    rows = model.row_degree <= cfg.seed_degree
    found: list[torch.Tensor] = []
    tried = 0
    while tried < cfg.n_seed_inits:
        tried += 1
        u = torch.zeros(basis.dim, dtype=DT)
        noise = torch.randn(int(dofs.sum()), generator=gen, dtype=DT)
        u[dofs] = noise / noise.norm()
        x0 = _seed_scale(model, u, target, rows) * u
        sol = lm_solve(model, x0, target, dof_mask=dofs, row_mask=rows, iters=cfg.lm_iters)
        if sol.rel <= cfg.ok_tol:
            found.append(sol.x)
            if len(found) >= cfg.n_seed_branches:
                break
    return found, tried


def extend_branch(
    model: InvariantModel,
    x: torch.Tensor,
    target: torch.Tensor,
    ell: int,
    cfg: MarchConfig,
    gen: torch.Generator,
    n_inits: int,
) -> tuple[list[torch.Tensor], float, float | None]:
    """Consistent degree-``ell`` completions of ``x``, with the best frozen and polished residuals seen."""
    basis = model.basis
    dofs = basis.l_of_dof == ell
    rows = model.row_degree == ell
    dofs_upto = basis.l_of_dof <= ell
    rows_upto = model.row_degree <= ell
    scale = math.sqrt(float((x[basis.l_of_dof == ell - 1] ** 2).sum()))
    sols: list[torch.Tensor] = []
    best_frozen = math.inf
    best_polished: float | None = None
    for _ in range(n_inits):
        x0 = x.clone()
        noise = torch.randn(int(dofs.sum()), generator=gen, dtype=DT)
        x0[dofs] = scale * noise / noise.norm()
        sol = lm_solve(model, x0, target, dof_mask=dofs, row_mask=rows, iters=cfg.lm_iters)
        best_frozen = min(best_frozen, sol.rel)
        if sol.rel > cfg.frozen_accept_tol:
            continue
        cand, rel = sol.x, sol.rel
        if cfg.joint_polish:
            pol = lm_solve(
                model, cand, target, dof_mask=dofs_upto, row_mask=rows_upto, iters=cfg.lm_iters
            )
            best_polished = pol.rel if best_polished is None else min(best_polished, pol.rel)
            cand, rel = pol.x, pol.rel
        if rel > cfg.ok_tol:
            continue
        if _distinct(cand, sols, dofs_upto, scale):
            sols.append(cand)
        single = (
            cfg.single_solution_from_degree is not None and ell >= cfg.single_solution_from_degree
        )
        if sols and (single or len(sols) >= cfg.max_solutions_per_branch):
            break
    return sols, best_frozen, best_polished


def march(
    model: InvariantModel, target: torch.Tensor, cfg: MarchConfig, gen: torch.Generator
) -> tuple[list[torch.Tensor], list[DegreeLog], int, int]:
    """Run the full march; returns surviving branches, per-degree log, seed starts used, seed branches."""
    branches, tried = seed_branches(model, target, cfg, gen)
    n_seed = len(branches)
    log = [DegreeLog(cfg.seed_degree, n_seed, math.nan, None)]
    for ell in range(cfg.seed_degree + 1, model.basis.lmax + 1):
        if not branches:
            break
        attempts = [cfg.n_extend_inits] + (
            [cfg.n_extend_inits * cfg.retry_factor] if cfg.retry_factor else []
        )
        retried = False
        for n_inits in attempts:
            nxt: list[torch.Tensor] = []
            best_frozen, best_polished = math.inf, None
            for b in branches:
                sols, bf, bp = extend_branch(model, b, target, ell, cfg, gen, n_inits)
                scale = math.sqrt(float((b[model.basis.l_of_dof == ell - 1] ** 2).sum()))
                nxt.extend(
                    s for s in sols if _distinct(s, nxt, model.basis.l_of_dof <= ell, scale)
                )
                best_frozen = min(best_frozen, bf)
                if bp is not None:
                    best_polished = bp if best_polished is None else min(best_polished, bp)
            if nxt:
                break
            retried = True
        retried = retried and bool(nxt)
        log.append(DegreeLog(ell, len(nxt), best_frozen, best_polished, retried=retried))
        logger.info(
            '  degree %2d: %d alive (best frozen %.1e, polished %s)%s',
            ell,
            len(nxt),
            best_frozen,
            'n/a' if best_polished is None else f'{best_polished:.1e}',
            ' after retry' if retried else '',
        )
        branches = nxt[: cfg.beam_width]
    return branches, log, tried, n_seed


def solve_target(
    model: InvariantModel,
    aligner: MomentAligner,
    target_grid: torch.Tensor,
    meta: dict[str, int],
    cfg: MarchConfig,
    seed: int,
    aligned_tol: float,
) -> TargetResult:
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    x_t = model.coefficients(target_grid[None])[0]
    t_grid = model.grid(x_t[None])[0]
    target = model.features(x_t[None])[0]
    branches, log, tried, n_seed = march(model, target, cfg, gen)

    scores: list[BranchScore] = []
    finals: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in branches:
        sol = lm_solve(model, b, target, iters=cfg.final_lm_iters)
        aligned, rot = aligner.align(model.grid(sol.x[None])[0], t_grid)
        scores.append(BranchScore(sol.rel, model.full_bispectrum_rel(sol.x, x_t), aligned))
        finals.append((sol.x, rot))

    best = min(
        range(len(scores)),
        key=lambda i: (scores[i].feature_rel, scores[i].aligned_image_rel),
        default=None,
    )
    collisions = sum(
        s.feature_rel <= cfg.ok_tol and s.aligned_image_rel > aligned_tol for s in scores
    )
    chosen = scores[best] if best is not None else None
    return TargetResult(
        target=meta['target'],
        digit_idx=meta['digit_idx'],
        rot_idx=meta['rot_idx'],
        label=meta['label'],
        seed_inits_tried=tried,
        n_seed_branches=n_seed,
        degrees=log,
        branches=scores,
        feature_rel=chosen.feature_rel if chosen else None,
        full_bispectrum_rel=chosen.full_bispectrum_rel if chosen else None,
        aligned_image_rel=chosen.aligned_image_rel if chosen else None,
        recovered=bool(
            chosen and chosen.feature_rel <= cfg.ok_tol and chosen.aligned_image_rel <= aligned_tol
        ),
        collisions=collisions,
        seconds=time.perf_counter() - t0,
        x_target=x_t.tolist(),
        x_recovered=finals[best][0].tolist() if best is not None else None,
        rotation=finals[best][1].tolist() if best is not None else None,
    )


# --------------------------------------------------------------------------
# Targets: rebuilt exactly as reconstruct.py builds them
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    signal_source: str
    n_signals: int
    n_rotations: int
    seed: int
    lmax: int
    nlat: int
    nlon: int
    data_dir: str


def build_targets(
    spec: TargetSpec, device: torch.device
) -> list[tuple[dict[str, int], torch.Tensor]]:
    """``(meta, float64 grid)`` per target, in reconstruct.py's record order (signal-major)."""
    sht = RealSHT(
        spec.nlat,
        spec.nlon,
        lmax=spec.lmax + 1,
        mmax=spec.lmax + 1,
        grid='equiangular',
        norm='ortho',
    )
    isht = InverseRealSHT(
        spec.nlat,
        spec.nlon,
        lmax=spec.lmax + 1,
        mmax=spec.lmax + 1,
        grid='equiangular',
        norm='ortho',
    )
    sht, isht = sht.to(device), isht.to(device)
    if spec.signal_source == 'smnist':
        images, labels = load_digits(
            Path(spec.data_dir), spec.n_signals, spec.nlat, spec.nlon, seed=spec.seed
        )
    elif spec.signal_source == 'random':
        images, labels = make_random_bandlimited_signals(
            spec.n_signals, spec.lmax, isht, spec.seed, device
        )
    else:
        raise ValueError(f'unknown signal source {spec.signal_source}')
    out: list[tuple[dict[str, int], torch.Tensor]] = []
    for d in range(spec.n_signals):
        with torch.no_grad():
            f = _bandlimit_project(
                images[d].to(device=device, dtype=torch.float32)[None], sht, isht
            )[0]
            stack, _ = make_rotation_stack(f, spec.n_rotations, seed=spec.seed * 7919 + d)
        for k in range(stack.shape[0]):
            meta = {'target': len(out), 'digit_idx': d, 'rot_idx': k, 'label': int(labels[d])}
            out.append((meta, stack[k].detach().cpu().to(DT)))
    return out


def check_against_state(
    targets: list[tuple[dict[str, int], torch.Tensor]], state_path: Path
) -> float:
    """Largest relative difference between rebuilt targets and those of a reconstruct.py run."""
    main = sys.modules['__main__']
    if not hasattr(main, 'ReconResult'):
        main.ReconResult = ReconResult  # state.pt was pickled from reconstruct.py run as __main__
    cached = {
        (r.digit_idx, r.rot_idx): r.target.to(DT)
        for r in torch.load(state_path, weights_only=False)['results']
    }
    worst = 0.0
    for meta, grid in targets:
        ref = cached[(meta['digit_idx'], meta['rot_idx'])]
        worst = max(worst, float((grid - ref).norm() / ref.norm()))
    return worst


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _shard_path(output_dir: Path, ids: list[int]) -> Path:
    return output_dir / 'shards' / f'targets_{"_".join(map(str, ids))}.json'


def cmd_solve(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.n_threads)
    spec = TargetSpec(
        args.signal_source,
        args.n_signals,
        args.n_rotations,
        args.seed,
        args.lmax,
        args.nlat,
        args.nlon,
        str(args.data_dir),
    )
    cfg = LEGACY_CONFIG if args.legacy else MarchConfig(seed_degree=args.seed_degree)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    targets = build_targets(spec, device)
    if args.adam_state is not None and args.adam_state.exists():
        logger.info(
            'targets vs %s: max rel diff %.2e',
            args.adam_state,
            check_against_state(targets, args.adam_state),
        )
    ids = args.targets if args.targets else list(range(len(targets)))

    model = InvariantModel(args.lmax, args.nlat, args.nlon)
    logger.info('Jacobian check: rel diff %.1e vs autodiff', model.check_jacobian())
    aligner = MomentAligner(model)

    path = _shard_path(args.output_dir, ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for i in ids:
        meta, grid = targets[i]
        logger.info(
            'target %d (digit %d, label %d, rot %d)',
            i,
            meta['digit_idx'],
            meta['label'],
            meta['rot_idx'],
        )
        res = solve_target(
            model, aligner, grid, meta, cfg, args.solver_seed * 100_003 + i, args.aligned_tol
        )
        logger.info(
            'target %d: %s | feature %s, full %s, aligned %s, %d branches, %d collisions (%.0fs)',
            i,
            'RECOVERED' if res.recovered else 'failed',
            _fmt(res.feature_rel),
            _fmt(res.full_bispectrum_rel),
            _fmt(res.aligned_image_rel, '.4f'),
            len(res.branches),
            res.collisions,
            res.seconds,
        )
        rows.append(asdict(res))
        path.write_text(json.dumps({'spec': asdict(spec), 'config': asdict(cfg), 'records': rows}))
    logger.info('wrote %s', path)


def _fmt(v: float | None, spec: str = '.1e') -> str:
    return 'n/a' if v is None else format(v, spec)


def cmd_merge(args: argparse.Namespace) -> None:
    shards = sorted((args.output_dir / 'shards').glob('targets_*.json'))
    if not shards:
        raise SystemExit(f'no shards in {args.output_dir / "shards"}')
    payloads = [json.loads(p.read_text()) for p in shards]
    spec, cfg = payloads[0]['spec'], payloads[0]['config']
    for p in payloads[1:]:
        if p['spec'] != spec or p['config'] != cfg:
            raise SystemExit('shards disagree on target spec or solver config')
    records = sorted((r for p in payloads for r in p['records']), key=lambda r: r['target'])
    if len({r['target'] for r in records}) != len(records):
        raise SystemExit('a target appears in two shards')
    n = len(records)
    summary = {
        'n_targets': n,
        'n_recovered': sum(r['recovered'] for r in records),
        'n_collisions': sum(r['collisions'] for r in records),
        'aligned_image_rel_recovered': _stats(
            [r['aligned_image_rel'] for r in records if r['recovered']]
        ),
        'feature_rel_recovered': _stats([r['feature_rel'] for r in records if r['recovered']]),
        'failed_targets': [r['target'] for r in records if not r['recovered']],
        'seconds_total': sum(r['seconds'] for r in records),
    }
    out = args.output_dir / 'results.json'
    out.write_text(
        json.dumps({'spec': spec, 'config': cfg, 'summary': summary, 'records': records}, indent=1)
        + '\n'
    )
    print(json.dumps(summary, indent=1))
    print(f'wrote {out}')


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values)
    return {'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


def records_to_results(payload: dict[str, object]) -> list[ReconResult]:
    """Rebuild reconstruct.py records (band-limited target, recovery, aligned recovery) from coefficients."""
    spec = payload['spec']
    model_basis = RealSHBasis.build(spec['lmax'])
    isht = InverseRealSHT(
        spec['nlat'],
        spec['nlon'],
        lmax=spec['lmax'] + 1,
        mmax=spec['lmax'] + 1,
        grid='equiangular',
        norm='ortho',
    ).to(DT)
    sht = RealSHT(
        spec['nlat'],
        spec['nlon'],
        lmax=spec['lmax'] + 1,
        mmax=spec['lmax'] + 1,
        grid='equiangular',
        norm='ortho',
    ).to(DT)
    out: list[ReconResult] = []
    for r in payload['records']:
        if r['x_recovered'] is None:
            continue
        x_t = torch.tensor(r['x_target'], dtype=DT)
        x_r = torch.tensor(r['x_recovered'], dtype=DT)
        target = isht(model_basis.to_coeffs(x_t[None]))[0]
        recovered = isht(model_basis.to_coeffs(x_r[None]))[0]
        rot = torch.tensor(r['rotation'], dtype=DT)
        aligned = isht(sht(rotate_spherical_function(recovered[None], rot)))[0]
        out.append(
            ReconResult(
                digit_idx=r['digit_idx'],
                rot_idx=r['rot_idx'],
                label=r['label'],
                invariance_rel=0.0,
                final_recon_rel=r['feature_rel'],
                image_space_rel=float((recovered - target).norm() / target.norm()),
                aligned_image_space_rel=r['aligned_image_rel'],
                seconds=r['seconds'],
                align_seconds=0.0,
                target=target.float(),
                initial=torch.zeros_like(target).float(),
                recovered=recovered.float(),
                aligned=aligned.float(),
            )
        )
    return out


def cmd_plot(args: argparse.Namespace) -> None:
    payload = json.loads(args.results.read_text())
    results = records_to_results(payload)
    make_paper_figure(
        results, digit_indices=args.paper_digits, path=args.paper_figure_path, rot_idx=args.rot_idx
    )
    if args.orbits_path is not None:
        spec = payload['spec']
        make_orbits_figure(
            results,
            n_digits=spec['n_signals'],
            n_rotations=spec['n_rotations'],
            path=args.orbits_path,
        )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='command', required=True)

    solve = sub.add_parser('solve', help='March a set of targets; writes one shard JSON.')
    solve.add_argument('--signal_source', choices=['smnist', 'random'], default='smnist')
    solve.add_argument('--n_signals', type=int, default=8)
    solve.add_argument('--n_rotations', type=int, default=1)
    solve.add_argument('--seed', type=int, default=2026, help='Target seed, as in reconstruct.py.')
    solve.add_argument('--lmax', type=int, default=15)
    solve.add_argument('--nlat', type=int, default=128)
    solve.add_argument('--nlon', type=int, default=256)
    solve.add_argument(
        '--data_dir', type=Path, default=here.parent / 'spherical_mnist' / 'smnist_data'
    )
    solve.add_argument(
        '--targets', type=int, nargs='*', default=None, help='Target indices (default: all).'
    )
    solve.add_argument('--seed_degree', type=int, default=4)
    solve.add_argument(
        '--legacy', action='store_true', help='Frozen marching without joint polish or retries.'
    )
    solve.add_argument('--solver_seed', type=int, default=0)
    solve.add_argument('--aligned_tol', type=float, default=0.05)
    solve.add_argument(
        '--adam_state',
        type=Path,
        default=None,
        help='reconstruct.py state.pt to check targets against.',
    )
    solve.add_argument('--n_threads', type=int, default=4)
    solve.add_argument('--output_dir', type=Path, required=True)

    merge = sub.add_parser('merge', help='Combine shards into results.json with a summary.')
    merge.add_argument('--output_dir', type=Path, required=True)

    plot = sub.add_parser(
        'plot', help='Paper figure (and optional orbit contact sheet) from results.json.'
    )
    plot.add_argument('--results', type=Path, required=True)
    plot.add_argument('--paper_figure_path', type=Path, required=True)
    plot.add_argument('--paper_digits', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    plot.add_argument('--rot_idx', type=int, default=0)
    plot.add_argument('--orbits_path', type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'
    )
    args = parse_args()
    {'solve': cmd_solve, 'merge': cmd_merge, 'plot': cmd_plot}[args.command](args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
