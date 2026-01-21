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

import subprocess
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
DT_COARSE = 0.1  # Snapshot interval
T_TOTAL = 5.0  # Total simulation time
R_PARAM = 1.0  # Instability parameter (strength of pattern growth)
L0_TARGET = 2  # Target degree for instability (l=2 modes grow)
OMEGA = 2.0  # Rotation speed (radians per unit time)
LAM = 0.1  # Bispectrum weight in lift
GAMMA = 1e-4  # Frobenius regularization
LR = 0.01  # Adam learning rate
TRAIN_STEPS = 500  # Koopman training iterations
NUM_SUBSTEPS = 10  # Substeps between coarse snapshots
EPS = 0.1  # sinθ clamping (larger to avoid pole instabilities)


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

    Returns:
        (N, N) learned generator matrix
    """
    N = lifts[0].shape[0]
    L = torch.zeros(N, N, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([L], lr=lr)

    # Stack pairs: (Phi_t, Phi_{t+dt})
    Phi_t = torch.stack([lift.float() for lift in lifts[:-1]])  # (T-1, N)
    Phi_next = torch.stack([lift.float() for lift in lifts[1:]])  # (T-1, N)

    initial_loss = None
    for step in range(steps):
        optimizer.zero_grad()

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

    final_loss = loss.item()
    print(f'  Training complete: initial={initial_loss:.6e} -> final={final_loss:.6e}')
    if final_loss > initial_loss / 10:
        print('  Warning: Loss did not decrease by 10x. Consider more steps or tuning.')

    return L.detach()


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
        subprocess.run(cmd, check=True, capture_output=True)
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
    print('Toy Koopman Bispectrum Demo')
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
    isht = InverseRealSHT(NLAT, NLON, lmax=LMAX + 1, mmax=LMAX + 1, grid='equiangular', norm='ortho')
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
    f_test = (torch.cos(theta) + 0.3 * torch.sin(theta) ** 2 * torch.cos(2 * phi)).squeeze().double()
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
    print(f'  f0 shape: {f0_grid.shape}, range: [{f0_grid.min().item():.3f}, {f0_grid.max().item():.3f}]')

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

    # --- Build lifts ---
    print('\n[5] Building lifted feature vectors...')
    lifts: list[torch.Tensor] = []
    for snap in snapshots:
        coeffs = grid_to_sh(snap.double(), sht)
        lift = build_lift(coeffs, bsp_module, LAM)
        lifts.append(lift)

    lift_dim = lifts[0].shape[0]
    print(f'  Lift dimension: {lift_dim}')

    # --- Train Koopman generator ---
    print('\n[6] Training Koopman generator L...')
    L = train_koopman_generator(lifts, DT_COARSE, GAMMA, LR, TRAIN_STEPS, device)
    print(f'  L shape: {L.shape}, ||L||_F = {torch.norm(L).item():.4f}')

    # --- Substep predictions ---
    print('\n[7] Generating substep predictions...')
    # Pick a coarse interval in the middle for visualization
    mid_idx = len(snapshots) // 2
    phi_start = lifts[mid_idx].float()

    substep_lifts = substep_rollout(phi_start, L, DT_COARSE, NUM_SUBSTEPS)

    # Decode lifts to spatial fields
    truth_fields: list[np.ndarray] = []
    pred_fields: list[np.ndarray] = []
    substep_times: list[float] = []

    t_start = coarse_times[mid_idx]
    delta = DT_COARSE / NUM_SUBSTEPS

    for j, lift_pred in enumerate(substep_lifts):
        # Predicted field
        coeffs_pred = lift_to_sh(lift_pred, LMAX, LMAX)
        f_pred = sh_to_grid(coeffs_pred, isht).detach().cpu().numpy()
        pred_fields.append(f_pred)

        # True field: simulate from mid_idx snapshot in spectral space
        t_sub = t_start + j * delta
        substep_times.append(t_sub)

        # Get true field at this substep time
        coeffs_sub = grid_to_sh(snapshots[mid_idx], sht)
        n_fine_steps = int(j * delta / DT_FINE)
        for _ in range(n_fine_steps):
            coeffs_sub = rk4_step_spectral(coeffs_sub, DT_FINE, rhs_fn)
        f_true = sh_to_grid(coeffs_sub, isht)
        truth_fields.append(f_true.detach().cpu().numpy())

    # --- Validation: intermediate prediction quality ---
    print('\n[8] Validating intermediate predictions...')
    mid_sub_idx = NUM_SUBSTEPS // 2
    truth_mid = truth_fields[mid_sub_idx]
    pred_mid = pred_fields[mid_sub_idx]
    mse_koopman = np.mean((pred_mid - truth_mid) ** 2)

    # Linear interpolation baseline
    f_start_np = snapshots[mid_idx].detach().cpu().numpy()
    f_end_np = snapshots[mid_idx + 1].detach().cpu().numpy()
    f_linear = 0.5 * (f_start_np + f_end_np)
    mse_linear = np.mean((f_linear - truth_mid) ** 2)

    print(f'  Midpoint MSE (Koopman): {mse_koopman:.6e}')
    print(f'  Midpoint MSE (linear):  {mse_linear:.6e}')
    if mse_koopman < mse_linear:
        print('  Koopman beats linear interpolation!')
    else:
        print('  Note: Linear interpolation is competitive (normal for smooth dynamics)')

    # --- Static comparison plot ---
    print('\n[9] Saving static comparison plot (sphere)...')
    static_path = out_dir / f'comparison_t{substep_times[mid_sub_idx]:.4f}.png'
    plot_comparison_sphere(truth_mid, pred_mid, substep_times[mid_sub_idx], static_path)
    print(f'  Saved: {static_path}')

    # --- Animation frames ---
    print('\n[10] Saving animation frames...')
    save_animation_frames(truth_fields, pred_fields, substep_times, frames_dir)

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
