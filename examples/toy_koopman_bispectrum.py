"""Toy Koopman bispectrum demo: learning continuous-time dynamics on the sphere.

This example demonstrates:
1. Simulating the advected Swift-Hohenberg equation on S²:
       ∂_t f = r*f - (1 + ∇²)²f - f³ + Ω ∂_φ f
   - Pattern formation from random noise
   - Rotation via exact spectral advection (i*m*Ω term)
   - All derivatives computed spectrally for numerical stability
2. Building lifted features: Φ(f) = [SH coeffs; λ * bispectrum(coeffs)]
3. Learning a Koopman generator L from coarse snapshots
4. Producing substep predictions via exp(L*j*δ)Φ_t
5. 3D sphere visualization of rotating patterns

Run from repo root:
    python examples/toy_koopman_bispectrum.py
    # or
    python -m examples.toy_koopman_bispectrum
"""

from __future__ import annotations

import subprocess  # nosec B404 - used for local ffmpeg invocation with fixed args
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_harmonics import InverseRealSHT, RealSHT

from bispectrum import SO3onS2

if TYPE_CHECKING:
    pass

# =============================================================================
# Hyperparameters
# =============================================================================
NLAT = 32  # Latitude points
NLON = 64  # Longitude points
LMAX = 5  # Max SH degree (CG limit)
# Advected Swift-Hohenberg parameters
DT_FINE = 5e-4  # Time step (larger is safe with spectral methods)
DT_COARSE = 0.05  # Snapshot interval (reduced for more training pairs)
T_TOTAL = 8.0  # Total simulation time (includes growth + saturation)
T_TRAIN_START = 3.5  # Start training from saturation regime (patterns formed)
R_PARAM = 1.0  # Instability parameter (strength of pattern growth)
L0_TARGET = 2  # Target degree for instability (l=2 modes grow)
OMEGA = 2.0  # Rotation speed (radians per unit time)
LAM = 1.0  # Bispectrum weight in lift (equal weight to SH coefficients)
GAMMA = 1e-3  # Frobenius regularization
LR = 0.01  # Adam learning rate
TRAIN_STEPS = 500  # Koopman training iterations
NUM_SUBSTEPS = 10  # Substeps between coarse snapshots
EPS = 0.1  # sinθ clamping (larger to avoid pole instabilities)
PCA_DIM = 50  # Dimension for PCA projection of full lift (reduces overfitting)


# =============================================================================
# Grid Construction
# =============================================================================
def make_sphere_grid(
    nlat: int, nlon: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Create equiangular grid on the sphere matching torch-harmonics convention.

    Args:
        nlat: Number of latitude points
        nlon: Number of longitude points
        device: Torch device

    Returns:
        theta: (nlat, 1) colatitude values in [0, pi]
        phi: (1, nlon) longitude values in [0, 2pi)
        sin_theta: (nlat, 1) sin(theta) clamped for safety
        dtheta: Grid spacing in theta
        dphi: Grid spacing in phi
    """
    # Match torch-harmonics equiangular grid convention
    theta_1d = torch.linspace(0, np.pi, nlat, device=device, dtype=torch.float64)
    phi_1d = torch.linspace(0, 2 * np.pi, nlon + 1, device=device, dtype=torch.float64)[:-1]

    theta = theta_1d.unsqueeze(1)  # (nlat, 1)
    phi = phi_1d.unsqueeze(0)  # (1, nlon)

    # sin(theta) with clamping for numerical safety at poles (theta=0 or pi)
    sin_theta = torch.clamp(torch.sin(theta), min=EPS)

    dtheta = np.pi / (nlat - 1)  # Note: nlat-1 intervals
    dphi = 2 * np.pi / nlon

    return theta, phi, sin_theta, dtheta, dphi


# =============================================================================
# Advected Swift-Hohenberg PDE (Rotating Pattern Formation)
# Equation: ∂t f = r*f - (l₀(l₀+1) + ∇²)² f - f³ + Ω ∂_φ f
# =============================================================================
def advected_sh_rhs(
    coeffs: torch.Tensor,
    sht: RealSHT,
    isht: InverseRealSHT,
    l_lap: torch.Tensor,
    m_vec: torch.Tensor,
    r_param: float = 1.0,
    l0_target: int = 2,
    omega: float = 2.0,
) -> torch.Tensor:
    """Compute RHS for advected Swift-Hohenberg in spectral space.

    Equation: ∂t f = r*f - (l₀(l₀+1) + ∇²)² f - f³ + Ω ∂_φ f

    This targets l=l0_target modes for instability. The operator becomes:
        r - (l₀(l₀+1) - l(l+1))²
    which is maximized (= r) when l = l0_target.

    The rotation term Ω ∂_φ f is computed exactly in spectral space as (i*m*Ω)*f_lm.

    Args:
        coeffs: (1, L, M) complex SH coefficients
        sht: Forward spherical harmonic transform
        isht: Inverse spherical harmonic transform
        l_lap: (L, M) precomputed Laplacian eigenvalues [-l(l+1)]
        m_vec: (L, M) precomputed m indices for rotation
        r_param: Instability parameter (positive = patterns form)
        l0_target: Target degree for instability (default l=2)
        omega: Rotation speed (radians per unit time)

    Returns:
        (1, L, M) RHS in spectral space
    """
    # Target eigenvalue for instability
    k0_sq = l0_target * (l0_target + 1)  # For l=2: k0_sq = 6

    # 1. Linear pattern term (spectral)
    # Operator: r - (k0² + ∇²)² = r - (k0² - l(l+1))²
    # l_lap = -l(l+1), so k0² + l_lap = k0² - l(l+1)
    linear_op = r_param - (k0_sq + l_lap) ** 2
    term_pattern = linear_op * coeffs

    # 2. Advection/rotation term (spectral)
    # ∂_φ corresponds to multiplying by i*m
    term_advection = 1j * m_vec * omega * coeffs

    # 3. Nonlinear term: -f³ (must compute in grid space)
    f_grid = isht(coeffs)
    nonlinear_grid = -(f_grid**3)
    term_nonlinear = sht(nonlinear_grid)

    return term_pattern + term_advection + term_nonlinear


def rk4_step_spectral(
    coeffs: torch.Tensor,
    dt: float,
    rhs_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Single RK4 time step in spectral space.

    Args:
        coeffs: Current SH coefficients
        dt: Time step size
        rhs_fn: Function computing the RHS in spectral space

    Returns:
        Updated SH coefficients
    """
    k1 = rhs_fn(coeffs)
    k2 = rhs_fn(coeffs + 0.5 * dt * k1)
    k3 = rhs_fn(coeffs + 0.5 * dt * k2)
    k4 = rhs_fn(coeffs + dt * k3)
    return coeffs + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# SHT Wrappers
# =============================================================================
def grid_to_sh(f: torch.Tensor, sht: RealSHT) -> torch.Tensor:
    """Convert grid field to spherical harmonic coefficients.

    Args:
        f: (nlat, nlon) real field
        sht: RealSHT transform object

    Returns:
        (1, lmax+1, mmax+1) complex SH coefficients
    """
    return sht(f.unsqueeze(0).double())


def sh_to_grid(coeffs: torch.Tensor, isht: InverseRealSHT) -> torch.Tensor:
    """Convert SH coefficients to grid field.

    Args:
        coeffs: (1, lmax+1, mmax+1) complex SH coefficients
        isht: InverseRealSHT transform object

    Returns:
        (nlat, nlon) real field
    """
    return isht(coeffs).squeeze(0)


# =============================================================================
# Lift Construction
# =============================================================================
def build_lift(
    coeffs: torch.Tensor,
    bsp_module: SO3onS2,
    lam: float,
) -> torch.Tensor:
    """Build lifted feature vector from SH coefficients.

    Φ(f) = [Re(coeffs_flat); Im(coeffs_flat); λ*Re(bsp); λ*Im(bsp)]

    Args:
        coeffs: (1, lmax+1, mmax+1) complex SH coefficients
        bsp_module: SO3onS2 bispectrum module
        lam: Weight for bispectrum features

    Returns:
        (N,) real feature vector
    """
    # Flatten SH coefficients
    c_flat = coeffs.flatten()  # complex
    sh_part = torch.cat([c_flat.real, c_flat.imag])

    # Compute bispectrum
    bsp = bsp_module(coeffs)  # (1, bsp_size) complex
    bsp_flat = bsp.flatten()
    bsp_part = lam * torch.cat([bsp_flat.real, bsp_flat.imag])

    return torch.cat([sh_part, bsp_part])


def build_lift_no_bispectrum(coeffs: torch.Tensor) -> torch.Tensor:
    """Build lifted feature vector using ONLY SH coefficients (no bispectrum).

    This serves as a baseline to compare against the full bispectrum lift.

    Args:
        coeffs: (1, L, M) complex SH coefficients

    Returns:
        (2*L*M,) real feature vector [Re(coeffs), Im(coeffs)]
    """
    c_flat = coeffs.flatten()  # complex
    return torch.cat([c_flat.real, c_flat.imag])


def lift_to_sh(
    lift: torch.Tensor,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    """Extract SH coefficients from lift vector (for decoding).

    Args:
        lift: (N,) feature vector
        lmax: Maximum l degree
        mmax: Maximum m degree

    Returns:
        (1, lmax+1, mmax+1) complex SH coefficients
    """
    num_sh = (lmax + 1) * (mmax + 1)
    real_part = lift[:num_sh]
    imag_part = lift[num_sh : 2 * num_sh]
    coeffs = torch.complex(real_part, imag_part)
    return coeffs.reshape(1, lmax + 1, mmax + 1)


# =============================================================================
# Koopman Training
# =============================================================================
def train_koopman_generator(
    lifts: list[torch.Tensor],
    dt_coarse: float,
    gamma: float,
    lr: float,
    steps: int,
    device: torch.device,
    antisymmetric: bool = True,
) -> torch.Tensor:
    """Learn Koopman generator L via gradient descent.

    Minimizes: ||Φ_{t+Δ} - exp(L*Δ)Φ_t||² + γ||L||_F²

    Args:
        lifts: List of (N,) lift vectors at consecutive coarse times
        dt_coarse: Time interval between consecutive lifts
        gamma: Frobenius regularization weight
        lr: Learning rate
        steps: Number of optimization steps
        device: Torch device
        antisymmetric: If True, constrain L to be antisymmetric (L = -L^T),
            which guarantees bounded (oscillatory) dynamics with no growth/decay.

    Returns:
        (N, N) learned generator matrix
    """
    N = lifts[0].shape[0]

    if antisymmetric:
        # Parameterize L = A - A^T (antisymmetric, eigenvalues are purely imaginary)
        A = torch.zeros(N, N, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([A], lr=lr)
    else:
        L_raw = torch.zeros(N, N, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([L_raw], lr=lr)

    # Stack pairs: (Phi_t, Phi_{t+dt})
    Phi_t = torch.stack([lift.float() for lift in lifts[:-1]])  # (T-1, N)
    Phi_next = torch.stack([lift.float() for lift in lifts[1:]])  # (T-1, N)

    initial_loss = None
    for step in range(steps):
        optimizer.zero_grad()

        # Construct L (antisymmetric if requested)
        if antisymmetric:
            L = A - A.T
        else:
            L = L_raw

        # exp(L*dt) @ Phi_t^T -> (N, T-1), then transpose
        expLdt = torch.matrix_exp(L * dt_coarse)
        pred = (expLdt @ Phi_t.T).T  # (T-1, N)

        loss_fit = torch.mean((pred - Phi_next) ** 2)
        loss_reg = gamma * torch.sum(L**2)
        loss = loss_fit + loss_reg

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f'  Step {step:4d}: loss={loss.item():.6e} (fit={loss_fit.item():.6e})')

    # Final L
    if antisymmetric:
        L = (A - A.T).detach()
    else:
        L = L_raw.detach()

    final_loss = loss.item()
    print(f'  Training complete: initial={initial_loss:.6e} -> final={final_loss:.6e}')
    if final_loss > initial_loss / 10:
        print('  Warning: Loss did not decrease by 10x. Consider more steps or tuning.')

    return L


def koopman_predict(phi_t: torch.Tensor, L: torch.Tensor, dt: float) -> torch.Tensor:
    """Predict lifted state at time t+dt using Koopman generator.

    Args:
        phi_t: (N,) current lift vector
        L: (N, N) generator matrix
        dt: Time step

    Returns:
        (N,) predicted lift vector
    """
    expLdt = torch.matrix_exp(L * dt)
    return expLdt @ phi_t


def substep_rollout(
    phi_start: torch.Tensor,
    L: torch.Tensor,
    dt_coarse: float,
    num_substeps: int,
) -> list[torch.Tensor]:
    """Generate substep predictions between coarse snapshots.

    Args:
        phi_start: (N,) starting lift vector
        L: (N, N) generator matrix
        dt_coarse: Coarse time interval
        num_substeps: Number of substeps

    Returns:
        List of (N,) lift vectors at substep times
    """
    delta = dt_coarse / num_substeps
    preds = []
    for j in range(num_substeps + 1):
        expLj = torch.matrix_exp(L * j * delta)
        phi_j = expLj @ phi_start
        preds.append(phi_j)
    return preds


# =============================================================================
# Visualization
# =============================================================================
def plot_sphere(
    field: np.ndarray,
    ax: plt.Axes,
    title: str,
    vmin: float,
    vmax: float,
    nlat: int,
    nlon: int,
) -> None:
    """Plot field on a 3D sphere.

    Args:
        field: (nlat, nlon) field values
        ax: Matplotlib 3D axes
        title: Plot title
        vmin, vmax: Colorbar limits
        nlat, nlon: Grid dimensions
    """
    # Create sphere coordinates
    theta_1d = np.linspace(0, np.pi, nlat)
    phi_1d = np.linspace(0, 2 * np.pi, nlon)
    phi_grid, theta_grid = np.meshgrid(phi_1d, theta_1d)

    # Convert to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Normalize field to [0, 1] for colormap
    norm_field = (field - vmin) / (vmax - vmin + 1e-10)
    norm_field = np.clip(norm_field, 0, 1)

    # Get colors from colormap
    cmap = plt.cm.RdBu_r
    colors = cmap(norm_field)

    # Plot surface
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, shade=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')


def plot_comparison_sphere(
    truth: np.ndarray,
    pred: np.ndarray,
    t: float,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Create 3-panel comparison plot with spheres: truth | predicted | error.

    Args:
        truth: (nlat, nlon) ground truth field
        pred: (nlat, nlon) predicted field
        t: Time value for title
        out_path: Output file path
        vmin, vmax: Colorbar limits (computed from data if None)
    """
    error = pred - truth
    mse = np.mean(error**2)
    nlat, nlon = truth.shape

    if vmin is None:
        vmin = min(truth.min(), pred.min())
    if vmax is None:
        vmax = max(truth.max(), pred.max())

    err_abs = max(np.abs(error).max(), 1e-10)

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(16, 5), dpi=100)

    # Truth sphere
    ax1 = fig.add_subplot(131, projection='3d')
    plot_sphere(truth, ax1, f'Truth (t={t:.4f})', vmin, vmax, nlat, nlon)

    # Prediction sphere
    ax2 = fig.add_subplot(132, projection='3d')
    plot_sphere(pred, ax2, 'Koopman Prediction', vmin, vmax, nlat, nlon)

    # Error sphere
    ax3 = fig.add_subplot(133, projection='3d')
    plot_sphere(error, ax3, f'Error (MSE={mse:.2e})', -err_abs, err_abs, nlat, nlon)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Field value')

    plt.suptitle(f'Spherical PDE Evolution (t={t:.4f})', fontsize=12, y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(
    truth: np.ndarray,
    pred: np.ndarray,
    t: float,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Create 3-panel comparison plot: truth | predicted | error.

    Args:
        truth: (nlat, nlon) ground truth field
        pred: (nlat, nlon) predicted field
        t: Time value for title
        out_path: Output file path
        vmin, vmax: Colorbar limits (computed from data if None)
    """
    error = pred - truth
    mse = np.mean(error**2)

    if vmin is None:
        vmin = min(truth.min(), pred.min())
    if vmax is None:
        vmax = max(truth.max(), pred.max())

    # Use figsize that produces even pixel dimensions for ffmpeg
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

    im0 = axes[0].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Truth (t={t:.4f})')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    im1 = axes[1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('Koopman Prediction')
    axes[1].set_xlabel('Longitude')

    err_abs = np.abs(error).max()
    im2 = axes[2].imshow(error, cmap='RdBu_r', vmin=-err_abs, vmax=err_abs, aspect='auto')
    axes[2].set_title(f'Error (MSE={mse:.2e})')
    axes[2].set_xlabel('Longitude')

    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_animation_frames(
    truth_fields: list[np.ndarray],
    pred_fields: list[np.ndarray],
    times: list[float],
    frames_dir: Path,
    use_sphere: bool = True,
) -> tuple[float, float]:
    """Save individual PNG frames for animation.

    Args:
        truth_fields: List of (nlat, nlon) ground truth fields
        pred_fields: List of (nlat, nlon) predicted fields
        times: List of time values
        frames_dir: Directory to save frames
        use_sphere: If True, use 3D sphere visualization

    Returns:
        (vmin, vmax) global colorbar limits used
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Compute global colorbar limits
    all_data = truth_fields + pred_fields
    vmin = min(f.min() for f in all_data)
    vmax = max(f.max() for f in all_data)

    plot_fn = plot_comparison_sphere if use_sphere else plot_comparison

    for i, (truth, pred, t) in enumerate(zip(truth_fields, pred_fields, times, strict=True)):
        frame_path = frames_dir / f'frame_{i:04d}.png'
        plot_fn(truth, pred, t, frame_path, vmin=vmin, vmax=vmax)

    print(f'Saved {len(times)} frames to {frames_dir}')
    return vmin, vmax


def save_animation_frames_enhanced(
    truth_fields: list[np.ndarray],
    pred_fields: list[np.ndarray],
    pred_fields_baseline: list[np.ndarray],
    times: list[float],
    frames_dir: Path,
    lift_dim_pca: int,
    lift_dim_full: int,
    lift_dim_baseline: int,
    pca_var_ratio: float,
    lmax: int,
) -> tuple[float, float]:
    """Save enhanced animation frames with metrics panels and explanations.

    Args:
        truth_fields: List of (nlat, nlon) ground truth fields
        pred_fields: List of (nlat, nlon) predicted fields (with bispectrum + PCA)
        pred_fields_baseline: List of (nlat, nlon) baseline predictions (no bispectrum)
        times: List of time values
        frames_dir: Directory to save frames
        lift_dim_pca: Dimension after PCA projection
        lift_dim_full: Original full lift dimension (before PCA)
        lift_dim_baseline: Dimension of the baseline lift (SH only)
        pca_var_ratio: Percentage of variance explained by PCA
        lmax: Maximum spherical harmonic degree

    Returns:
        (vmin, vmax) global colorbar limits used
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Compute global colorbar limits
    all_data = truth_fields + pred_fields + pred_fields_baseline
    vmin = min(f.min() for f in all_data)
    vmax = max(f.max() for f in all_data)

    # Precompute all MSEs and errors for the time series plot
    mses = []
    mses_baseline = []
    max_vals_truth = []
    max_vals_pred = []
    all_errors = []
    all_errors_baseline = []
    for truth, pred, pred_base in zip(
        truth_fields, pred_fields, pred_fields_baseline, strict=True
    ):
        error = pred - truth
        error_base = pred_base - truth
        mses.append(np.mean(error**2))
        mses_baseline.append(np.mean(error_base**2))
        max_vals_truth.append(np.abs(truth).max())
        max_vals_pred.append(np.abs(pred).max())
        all_errors.append(error)
        all_errors_baseline.append(error_base)

    # Global error scale for consistent normalization across all frames
    global_err_max = max(
        max(np.abs(e).max() for e in all_errors),
        max(np.abs(e).max() for e in all_errors_baseline),
        1e-10,
    )

    nlat, nlon = truth_fields[0].shape

    for i, (truth, pred, pred_base, t) in enumerate(
        zip(truth_fields, pred_fields, pred_fields_baseline, times, strict=True)
    ):
        frame_path = frames_dir / f'frame_{i:04d}.png'
        _plot_frame_enhanced(
            truth,
            pred,
            pred_base,
            t,
            times,
            mses,
            mses_baseline,
            max_vals_truth,
            max_vals_pred,
            i,
            frame_path,
            vmin,
            vmax,
            global_err_max,
            nlat,
            nlon,
            lift_dim_pca,
            lift_dim_full,
            lift_dim_baseline,
            pca_var_ratio,
            lmax,
        )

    print(f'Saved {len(times)} frames to {frames_dir}')
    return vmin, vmax


def _plot_frame_enhanced(
    truth: np.ndarray,
    pred: np.ndarray,
    pred_baseline: np.ndarray,
    t: float,
    all_times: list[float],
    all_mses: list[float],
    all_mses_baseline: list[float],
    max_vals_truth: list[float],
    max_vals_pred: list[float],
    frame_idx: int,
    out_path: Path,
    vmin: float,
    vmax: float,
    global_err_max: float,
    nlat: int,
    nlon: int,
    lift_dim_pca: int,
    lift_dim_full: int,
    lift_dim_baseline: int,
    pca_var_ratio: float,
    lmax: int,
) -> None:
    """Plot a single enhanced frame with spheres, metrics, and explanation."""
    error = pred - truth
    error_baseline = pred_baseline - truth
    mse = all_mses[frame_idx]
    mse_baseline = all_mses_baseline[frame_idx]
    # Use global error scale for consistent normalization across all frames
    err_abs = global_err_max

    # Create figure with GridSpec for complex layout (3 rows)
    fig = plt.figure(figsize=(22, 14), dpi=100)

    # Row 1: Truth + With Bispectrum + Error (with bisp)
    ax_truth = fig.add_subplot(3, 4, 1, projection='3d')
    ax_pred = fig.add_subplot(3, 4, 2, projection='3d')
    ax_error = fig.add_subplot(3, 4, 3, projection='3d')

    # Row 1: Baseline (no bisp) + Error (no bisp)
    ax_pred_base = fig.add_subplot(3, 4, 5, projection='3d')
    ax_error_base = fig.add_subplot(3, 4, 6, projection='3d')

    # Plot spheres - Row 1
    plot_sphere(truth, ax_truth, 'Ground Truth (PDE)', vmin, vmax, nlat, nlon)
    plot_sphere(pred, ax_pred, f'Bispectrum+PCA (MSE={mse:.2e})', vmin, vmax, nlat, nlon)
    plot_sphere(error, ax_error, 'Error (Bisp+PCA)', -err_abs, err_abs, nlat, nlon)

    # Plot spheres - Row 2 (baseline)
    plot_sphere(
        pred_baseline, ax_pred_base, f'SH Only (MSE={mse_baseline:.2e})', vmin, vmax, nlat, nlon
    )
    plot_sphere(error_baseline, ax_error_base, 'Error (SH Only)', -err_abs, err_abs, nlat, nlon)

    # Top right: Model explanation
    ax_text = fig.add_subplot(3, 4, 4)
    ax_text.axis('off')
    improvement = (mse_baseline - mse) / mse_baseline * 100 if mse_baseline > 0 else 0

    # Compute irrep stats for lmax
    n_sh_modes = (lmax + 1) ** 2  # Total SH modes (all m)
    n_l_pairs = (lmax + 1) * (lmax + 2) // 2  # (l1, l2) pairs with l1 <= l2
    n_bisp = lift_dim_full - lift_dim_baseline  # Bispectrum features (real+imag)

    explanation = (
        r'$\bf{SO(3)\ Bispectrum\ on\ S^2}$'
        '\n\n'
        r'$\bf{PDE:}$ Advected Swift-Hohenberg'
        '\n'
        r'$\partial_t f = rf - (1+\nabla^2)^2 f - f^3 + \Omega \partial_\phi f$'
        '\n\n'
        r'$\bf{Irrep\ Statistics:}$'
        '\n'
        f'• $\\ell_{{max}}$={lmax}, SH modes: {n_sh_modes}\n'
        f'• $(\\ell_1,\\ell_2)$ pairs: {n_l_pairs}\n'
        f'• Bispectrum values: {n_bisp // 2} complex\n'
        f'• Improvement: {improvement:.1f}%'
    )
    ax_text.text(
        0.05,
        0.95,
        explanation,
        transform=ax_text.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8},
    )

    # Middle right: Method comparison box
    ax_method = fig.add_subplot(3, 4, 8)
    ax_method.axis('off')
    method_text = (
        r'$\bf{SO(3)\ Bispectrum:}$'
        '\n'
        r'$B_{\ell_1\ell_2}^\ell = (F_{\ell_1} \otimes F_{\ell_2}) \cdot C_{\ell_1\ell_2}^\ell \cdot F_\ell^*$'
        '\n'
        '  Clebsch-Gordan coupling\n'
        '  SO(3)-invariant features\n'
        '\n'
        r'$\bf{Lift\ Dimensions:}$'
        '\n'
        f'  Full: {lift_dim_full}→{lift_dim_pca} (PCA)\n'
        f'  Baseline: {lift_dim_baseline} (SH only)\n'
        '\n'
        r'$\bf{Koopman:}$ $\Phi_{t+\Delta} = e^{L\Delta} \Phi_t$'
        '\n'
        '  L: antisymmetric generator\n'
        '  One-step predictions'
    )
    ax_method.text(
        0.05,
        0.95,
        method_text,
        transform=ax_method.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.8},
    )

    # Bottom row: MSE comparison plot
    ax_mse = fig.add_subplot(3, 4, 9)
    ax_mse.plot(all_times, all_mses, 'b-', linewidth=2.5, label='With Bispectrum')
    ax_mse.plot(all_times, all_mses_baseline, 'r--', linewidth=2.5, label='No Bispectrum')
    ax_mse.axvline(t, color='gray', linestyle=':', linewidth=2)
    ax_mse.scatter([t], [mse], color='b', s=120, zorder=5, edgecolors='white', linewidth=2)
    ax_mse.scatter(
        [t], [mse_baseline], color='r', s=120, zorder=5, edgecolors='white', linewidth=2
    )
    ax_mse.set_xlabel('Time', fontsize=12)
    ax_mse.set_ylabel('MSE', fontsize=12)
    ax_mse.set_title('MSE Comparison: Bispectrum vs Baseline', fontsize=12, fontweight='bold')
    ax_mse.legend(loc='upper right', fontsize=10)
    ax_mse.grid(True, alpha=0.3)
    ax_mse.set_xlim(all_times[0], all_times[-1])
    ax_mse.set_ylim(0, max(max(all_mses), max(all_mses_baseline)) * 1.2)

    # Bottom: Improvement over time
    ax_imp = fig.add_subplot(3, 4, 10)
    improvements = [
        (mb - m) / mb * 100 if mb > 0 else 0
        for m, mb in zip(all_mses, all_mses_baseline, strict=True)
    ]
    ax_imp.fill_between(all_times, improvements, alpha=0.3, color='green')
    ax_imp.plot(all_times, improvements, 'g-', linewidth=2.5)
    ax_imp.axvline(t, color='gray', linestyle=':', linewidth=2)
    ax_imp.scatter(
        [t], [improvement], color='green', s=120, zorder=5, edgecolors='white', linewidth=2
    )
    ax_imp.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_imp.set_xlabel('Time', fontsize=12)
    ax_imp.set_ylabel('Improvement (%)', fontsize=12)
    ax_imp.set_title('Bispectrum Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax_imp.grid(True, alpha=0.3)
    ax_imp.set_xlim(all_times[0], all_times[-1])

    # Bottom: Flat projections
    ax_flat_truth = fig.add_subplot(3, 4, 11)
    im_flat = ax_flat_truth.imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax_flat_truth.set_title('Truth (Mercator)', fontsize=11)
    ax_flat_truth.set_xlabel('Longitude')
    ax_flat_truth.set_ylabel('Latitude')
    fig.colorbar(im_flat, ax=ax_flat_truth, shrink=0.7)

    ax_flat_pred = fig.add_subplot(3, 4, 12)
    im_flat2 = ax_flat_pred.imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax_flat_pred.set_title('With Bispectrum (Mercator)', fontsize=11)
    ax_flat_pred.set_xlabel('Longitude')
    fig.colorbar(im_flat2, ax=ax_flat_pred, shrink=0.7)

    # Main title
    plt.suptitle(
        f'Koopman with SO(3) Bispectrum on S²    |    t = {t:.3f}    |    '
        f'Bisp MSE: {mse:.2e}    |    Baseline MSE: {mse_baseline:.2e}',
        fontsize=14,
        fontweight='bold',
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


def run_ffmpeg(frames_dir: Path, out_mp4: Path) -> bool:
    """Stitch frames into MP4 using ffmpeg.

    Args:
        frames_dir: Directory containing frame_XXXX.png files
        out_mp4: Output MP4 file path

    Returns:
        True if successful, False otherwise
    """
    # Use -vf pad to ensure even dimensions (required by libx264)
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        '10',
        '-i',
        str(frames_dir / 'frame_%04d.png'),
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)  # nosec B603 - cmd is fixed
        print(f'Saved animation to {out_mp4}')
        return True
    except FileNotFoundError:
        print(f'ffmpeg not found. Frames saved to {frames_dir}')
        return False
    except subprocess.CalledProcessError as e:
        print(f'ffmpeg failed: {e.stderr.decode() if e.stderr else "unknown error"}')
        print(f'Frames saved to {frames_dir}')
        return False


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Run the toy Koopman bispectrum demo."""
    print('=' * 70)
    print('Koopman with SO(3) Bispectrum on S² Demo')
    print('=' * 70)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Output directory
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    frames_dir = out_dir / 'frames'

    # --- Setup ---
    print('\n[1] Setting up grid and transforms...')
    theta, phi, sin_theta, dtheta, dphi = make_sphere_grid(NLAT, NLON, device)

    # SHT transforms
    # torch-harmonics uses lmax/mmax as the number of modes (0 to lmax-1)
    # So for max degree LMAX, we need lmax=LMAX+1
    sht = RealSHT(NLAT, NLON, lmax=LMAX + 1, mmax=LMAX + 1, grid='equiangular', norm='ortho')
    isht = InverseRealSHT(
        NLAT, NLON, lmax=LMAX + 1, mmax=LMAX + 1, grid='equiangular', norm='ortho'
    )
    sht = sht.to(device).double()
    isht = isht.to(device).double()

    # Precompute Laplacian eigenvalues: l_lap[l, m] = -l(l+1)
    l_indices = torch.arange(LMAX + 1, device=device, dtype=torch.float64)
    l_lap = -l_indices * (l_indices + 1)
    l_lap = l_lap.view(-1, 1).expand(-1, LMAX + 1)  # (L, M)

    # Precompute m-values for rotation: m_vec[l, m] = m
    m_indices = torch.arange(LMAX + 1, device=device, dtype=torch.float64)
    m_vec = m_indices.view(1, -1).expand(LMAX + 1, -1)  # (L, M)

    # Bispectrum module
    bsp_module = SO3onS2(lmax=LMAX).to(device)
    print(f'  Bispectrum output size: {bsp_module.output_size}')

    # --- Validation: SHT round-trip ---
    print('\n[2] Validating SHT round-trip...')
    # Use function within bandwidth (l <= LMAX)
    f_test = (
        (torch.cos(theta) + 0.3 * torch.sin(theta) ** 2 * torch.cos(2 * phi)).squeeze().double()
    )
    coeffs_test = grid_to_sh(f_test, sht)
    f_recon = sh_to_grid(coeffs_test, isht).double()
    rel_error = torch.norm(f_recon - f_test) / torch.norm(f_test)
    print(f'  Relative error: {rel_error.item():.2e}')
    if rel_error > 1e-4:
        print('  Warning: SHT round-trip error is high!')

    # --- Initial condition ---
    print('\n[3] Setting initial condition (random noise for pattern formation)...')
    # Swift-Hohenberg grows patterns from small random perturbations
    torch.manual_seed(42)
    f0_grid = 0.2 * torch.randn(NLAT, NLON, device=device, dtype=torch.float64)

    # Convert to spectral space (simulation happens in spectral space)
    coeffs = grid_to_sh(f0_grid, sht)
    f0_grid = sh_to_grid(coeffs, isht)  # Bandlimited version
    print(
        f'  f0 shape: {f0_grid.shape}, range: [{f0_grid.min().item():.3f}, {f0_grid.max().item():.3f}]'
    )

    # --- PDE simulation ---
    print('\n[4] Simulating advected Swift-Hohenberg equation...')
    n_coarse = int(T_TOTAL / DT_COARSE) + 1
    steps_per_coarse = int(DT_COARSE / DT_FINE)

    def rhs_fn(c: torch.Tensor) -> torch.Tensor:
        return advected_sh_rhs(c, sht, isht, l_lap, m_vec, R_PARAM, L0_TARGET, OMEGA)

    # Collect snapshots (store grid fields for visualization)
    snapshots: list[torch.Tensor] = [f0_grid.clone()]
    coarse_times: list[float] = [0.0]

    # Time stepping in spectral space
    for i in range(1, n_coarse):
        # Fine time stepping (RK4 in spectral space)
        for _ in range(steps_per_coarse):
            coeffs = rk4_step_spectral(coeffs, DT_FINE, rhs_fn)

        # Convert to grid for snapshot storage and monitoring
        f_grid = sh_to_grid(coeffs, isht)
        f_max = torch.abs(f_grid).max().item()

        # Check for blowup
        if f_max > 100:
            print(f'  ERROR: PDE blew up at t={i * DT_COARSE:.3f}, max|f|={f_max:.1f}')
            return

        snapshots.append(f_grid.clone())
        coarse_times.append(i * DT_COARSE)

        if i % 5 == 0:
            print(f'  t={coarse_times[-1]:.2f}, max|f|={f_max:.3f}')

    print(f'  Collected {len(snapshots)} coarse snapshots')

    # --- Build lifts (with bispectrum and baseline without) ---
    print('\n[5] Building lifted feature vectors...')

    # Full lift: SH + bispectrum
    lifts: list[torch.Tensor] = []
    # Baseline lift: SH only (no bispectrum)
    lifts_baseline: list[torch.Tensor] = []

    for snap in snapshots:
        coeffs = grid_to_sh(snap.double(), sht)
        lift = build_lift(coeffs, bsp_module, LAM)
        lifts.append(lift)
        lift_baseline = build_lift_no_bispectrum(coeffs)
        lifts_baseline.append(lift_baseline)

    lift_dim_full = lifts[0].shape[0]
    lift_dim_baseline = lifts_baseline[0].shape[0]
    print(f'  Full lift dimension (SH + Bispectrum): {lift_dim_full}')
    print(f'  Baseline lift dimension (SH only): {lift_dim_baseline}')

    # --- PCA projection for full lifts (reduces overfitting) ---
    print('\n[5b] Applying PCA to full lifts...')
    lifts_stack = torch.stack(lifts, dim=0).float()  # [N, lift_dim_full]

    # Center the data
    lift_mean = lifts_stack.mean(dim=0)
    lifts_centered = lifts_stack - lift_mean

    # Compute PCA via SVD
    U, S, Vh = torch.linalg.svd(lifts_centered, full_matrices=False)
    # Keep top PCA_DIM components
    pca_dim = min(PCA_DIM, lift_dim_full, len(lifts))
    V_pca = Vh[:pca_dim, :].T  # [lift_dim_full, pca_dim]

    # Project lifts to PCA space
    lifts_pca = [((lift.float() - lift_mean) @ V_pca) for lift in lifts]

    # Compute variance explained
    total_var = (S**2).sum().item()
    explained_var = (S[:pca_dim] ** 2).sum().item()
    var_ratio = explained_var / total_var * 100

    print(f'  PCA: {lift_dim_full} -> {pca_dim} dims ({var_ratio:.1f}% variance explained)')
    lift_dim = pca_dim  # Use PCA dimension for Koopman

    # --- Train Koopman generators (on saturated regime only) ---
    print('\n[6] Training Koopman generators...')
    # Find index where training starts (after pattern saturation)
    train_start_idx = int(T_TRAIN_START / DT_COARSE)
    lifts_train = lifts_pca[train_start_idx:]  # Use PCA-projected lifts
    lifts_train_baseline = lifts_baseline[train_start_idx:]
    print(f'  Training on saturated regime: t >= {T_TRAIN_START} ({len(lifts_train)} snapshots)')

    print('  [6a] Training WITH bispectrum (PCA-projected)...')
    L = train_koopman_generator(lifts_train, DT_COARSE, GAMMA, LR, TRAIN_STEPS, device)
    print(f'       L shape: {L.shape}, ||L||_F = {torch.norm(L).item():.4f}')

    print('  [6b] Training WITHOUT bispectrum (baseline)...')
    L_baseline = train_koopman_generator(
        lifts_train_baseline, DT_COARSE, GAMMA, LR, TRAIN_STEPS, device
    )
    print(
        f'       L_baseline shape: {L_baseline.shape}, ||L||_F = {torch.norm(L_baseline).item():.4f}'
    )

    # --- Generate predictions for saturated timeline ---
    print('\n[7] Generating Koopman predictions for saturated regime...')

    # Generate predictions using ONE-STEP Koopman from TRUE lifts
    # This shows model accuracy without error compounding
    truth_fields: list[np.ndarray] = []
    pred_fields: list[np.ndarray] = []  # With bispectrum (PCA)
    pred_fields_baseline: list[np.ndarray] = []  # Without bispectrum
    frame_times: list[float] = []

    # Precompute exp(L * dt) for one-step prediction
    expLdt = torch.matrix_exp(L * DT_COARSE)
    expLdt_baseline = torch.matrix_exp(L_baseline * DT_COARSE)

    # Work with saturated regime snapshots only
    snapshots_sat = snapshots[train_start_idx:]
    times_sat = coarse_times[train_start_idx:]
    lifts_sat_pca = lifts_pca[train_start_idx:]  # PCA-projected full lifts
    lifts_sat_baseline = lifts_baseline[train_start_idx:]

    for i, (snap, t) in enumerate(zip(snapshots_sat, times_sat, strict=True)):
        # Truth: directly from simulation
        truth_fields.append(snap.detach().cpu().numpy())
        frame_times.append(t)

        # Prediction WITH bispectrum (PCA): one-step from TRUE previous lift
        if i == 0:
            pred_lift_pca = lifts_sat_pca[0].float()
            pred_lift_baseline = lifts_sat_baseline[0].float()
        else:
            # Predict from TRUE previous lift (in PCA space)
            prev_lift_pca = lifts_sat_pca[i - 1].float()
            pred_lift_pca = expLdt @ prev_lift_pca

            prev_lift_baseline = lifts_sat_baseline[i - 1].float()
            pred_lift_baseline = expLdt_baseline @ prev_lift_baseline

        # Decode full prediction: PCA space -> original space -> SH coeffs
        # Project back from PCA: pred_lift_full = pred_lift_pca @ V_pca.T + lift_mean
        pred_lift_full = pred_lift_pca @ V_pca.T + lift_mean
        coeffs_pred = lift_to_sh(pred_lift_full, LMAX, LMAX)
        f_pred = sh_to_grid(coeffs_pred, isht).detach().cpu().numpy()
        pred_fields.append(f_pred)

        # Decode baseline prediction
        coeffs_pred_baseline = lift_to_sh(pred_lift_baseline, LMAX, LMAX)
        f_pred_baseline = sh_to_grid(coeffs_pred_baseline, isht).detach().cpu().numpy()
        pred_fields_baseline.append(f_pred_baseline)

    print(
        f'  Generated {len(frame_times)} frames from t={frame_times[0]:.2f} to t={frame_times[-1]:.2f}'
    )

    # --- Validation: prediction quality over time ---
    print('\n[8] Validating predictions...')
    mid_idx = len(frame_times) // 2

    # With bispectrum
    truth_mid = truth_fields[mid_idx]
    pred_mid = pred_fields[mid_idx]
    mse_mid = np.mean((pred_mid - truth_mid) ** 2)
    mse_final = np.mean((pred_fields[-1] - truth_fields[-1]) ** 2)

    # Baseline (no bispectrum)
    pred_mid_baseline = pred_fields_baseline[mid_idx]
    mse_mid_baseline = np.mean((pred_mid_baseline - truth_mid) ** 2)
    mse_final_baseline = np.mean((pred_fields_baseline[-1] - truth_fields[-1]) ** 2)

    print('  WITH Bispectrum:')
    print(f'    MSE at t={frame_times[mid_idx]:.2f} (midpoint): {mse_mid:.6e}')
    print(f'    MSE at t={frame_times[-1]:.2f} (final): {mse_final:.6e}')
    print('  WITHOUT Bispectrum (baseline):')
    print(f'    MSE at t={frame_times[mid_idx]:.2f} (midpoint): {mse_mid_baseline:.6e}')
    print(f'    MSE at t={frame_times[-1]:.2f} (final): {mse_final_baseline:.6e}')
    improvement = (mse_final_baseline - mse_final) / mse_final_baseline * 100
    print(f'  Bispectrum improvement: {improvement:.1f}% lower MSE')

    # --- Static comparison plot ---
    print('\n[9] Saving static comparison plot (sphere)...')
    static_path = out_dir / f'comparison_t{frame_times[mid_idx]:.4f}.png'
    plot_comparison_sphere(truth_mid, pred_mid, frame_times[mid_idx], static_path)
    print(f'  Saved: {static_path}')

    # --- Animation frames (enhanced with metrics and baseline comparison) ---
    print('\n[10] Saving animation frames with metrics...')
    save_animation_frames_enhanced(
        truth_fields,
        pred_fields,
        pred_fields_baseline,
        frame_times,
        frames_dir,
        lift_dim,  # PCA dimension
        lift_dim_full,  # Original full dimension
        lift_dim_baseline,
        var_ratio,  # PCA variance explained
        LMAX,
    )

    # --- Create MP4 ---
    print('\n[11] Creating MP4 animation...')
    mp4_path = out_dir / 'koopman_animation.mp4'
    run_ffmpeg(frames_dir, mp4_path)

    # --- Summary ---
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  Outputs saved to: {out_dir.absolute()}')
    print(f'  - Static plot: {static_path.name}')
    print(f'  - Frames: {frames_dir.name}/')
    print(f'  - Animation: {mp4_path.name}')
    print(f'  Lift dimension: {lift_dim}')
    print(f'  Koopman generator: {L.shape[0]}x{L.shape[1]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
