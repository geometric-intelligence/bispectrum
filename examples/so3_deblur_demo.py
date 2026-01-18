import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch_harmonics import InverseRealSHT

from bispectrum import SO3onS2


def run_deblurring_demo():
    """
    Demo: Recovering high-frequency structure using bispectrum constraints.

    Scenario: We observe a blurry signal and want to reconstruct the sharp original.
    - Model A: Only regularization toward input (no structural prior) - stays blurry
    - Model B: Regularization + bispectrum constraint - recovers high-frequency structure

    The bispectrum acts as a "structural fingerprint" that encodes phase relationships
    between different frequency components, allowing recovery of detail that pure
    smoothness-based methods cannot achieve.
    """
    # --- Configuration ---
    LMAX = 5  # Resolution (keep low for speed)
    LAMBDA_REG = 0.01  # Regularization toward input (prevents divergence)
    LAMBDA_BSP = 10.0  # Strength of bispectrum constraint
    STEPS = 400

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsp_module = SO3onS2(lmax=LMAX).to(device)

    # --- 1. Generate "Ground Truth" (Sharp Signal) ---
    # Power law decay l^-2 with meaningful high-frequency content
    torch.manual_seed(42)  # Reproducibility
    truth_coeffs = torch.randn(1, LMAX + 1, LMAX + 1, dtype=torch.complex64, device=device)
    for l in range(LMAX + 1):
        truth_coeffs[:, l, :] *= 1.0 / (1.0 + l**2)

    # Pre-compute the "Structural Fingerprint" (rotation-invariant bispectrum)
    # In practice, this could come from physical constraints or a reference signal
    with torch.no_grad():
        truth_invariants = bsp_module(truth_coeffs)

    # --- 2. Generate "Input" (Blurry Observation) ---
    # Heavily dampen high frequencies to simulate degraded observation
    input_coeffs = truth_coeffs.clone().detach()
    for l in range(3, LMAX + 1):
        input_coeffs[:, l, :] *= 0.1  # Kill high frequencies

    # --- 3. Optimization: Regularization Only vs. Regularization + Bispectrum ---
    # Neither model has access to ground truth pixels!

    # Model A: Only stays close to input (no structural guidance)
    coeffs_A = input_coeffs.clone().detach().requires_grad_(True)
    opt_A = optim.Adam([coeffs_A], lr=0.01)

    # Model B: Stays close to input + matches bispectrum structure
    coeffs_B = input_coeffs.clone().detach().requires_grad_(True)
    opt_B = optim.Adam([coeffs_B], lr=0.01)

    print('Training starts...')
    print('Model A: Regularization only (no structural prior)')
    print('Model B: Regularization + Bispectrum constraint')
    print('-' * 50)

    for step in range(STEPS):
        # --- Model A: Only regularization toward input ---
        opt_A.zero_grad()
        # Just penalize deviation from input - will stay blurry
        loss_reg_A = torch.mean(torch.abs(coeffs_A - input_coeffs) ** 2)
        loss_A = loss_reg_A
        loss_A.backward()
        opt_A.step()

        # --- Model B: Regularization + Bispectrum ---
        opt_B.zero_grad()

        # 1. Regularization: don't deviate too far from input
        loss_reg_B = torch.mean(torch.abs(coeffs_B - input_coeffs) ** 2)

        # 2. Bispectrum constraint: match the structural fingerprint
        pred_invariants = bsp_module(coeffs_B)
        loss_bsp = torch.mean(torch.abs(pred_invariants - truth_invariants) ** 2)

        loss_B = LAMBDA_REG * loss_reg_B + LAMBDA_BSP * loss_bsp
        loss_B.backward()
        opt_B.step()

        if step % 100 == 0:
            print(
                f'Step {step}: Loss A={loss_A.item():.6f} | Loss B={loss_B.item():.6f} (bsp={loss_bsp.item():.6f})'
            )

    print('-' * 50)
    print('Done!')

    return truth_coeffs, input_coeffs, coeffs_A.detach(), coeffs_B.detach()


# --- Spherical Rendering ---
def sh_to_spatial(coeffs: torch.Tensor, nlat: int = 64, nlon: int = 128) -> np.ndarray:
    """Convert SH coefficients to a spatial grid on the sphere using torch-harmonics.

    Args:
        coeffs: (1, lmax+1, mmax+1) complex tensor with SH coefficients
        nlat: number of latitude points
        nlon: number of longitude points

    Returns:
        (nlat, nlon) real array of the function on the sphere
    """
    lmax = coeffs.shape[1]
    mmax = coeffs.shape[2]

    # Create inverse SHT
    isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid='equiangular', norm='ortho')
    isht = isht.to(coeffs.device)

    # Transform to spatial domain
    spatial = isht(coeffs.to(torch.complex64)).squeeze(0).detach().cpu().numpy()

    return spatial


# --- Analysis Helper ---
def analyze_results(truth, blurry, res_reg_only, res_bispectrum):
    """
    Combined visualization: spatial images + power spectrum.
    """

    def get_power_spectrum(coeffs):
        return torch.sum(torch.abs(coeffs) ** 2, dim=-1).squeeze().detach().cpu().numpy()

    # Render all signals to spatial domain
    print('Rendering spherical harmonics to spatial domain...')
    img_truth = sh_to_spatial(truth)
    img_blurry = sh_to_spatial(blurry)
    img_reg = sh_to_spatial(res_reg_only)
    img_bsp = sh_to_spatial(res_bispectrum)

    # Compute power spectra
    ps_truth = get_power_spectrum(truth)
    ps_blurry = get_power_spectrum(blurry)
    ps_reg = get_power_spectrum(res_reg_only)
    ps_bsp = get_power_spectrum(res_bispectrum)

    # Common colormap range for fair comparison
    vmin = min(img_truth.min(), img_blurry.min(), img_reg.min(), img_bsp.min())
    vmax = max(img_truth.max(), img_blurry.max(), img_reg.max(), img_bsp.max())

    # Create combined figure
    fig = plt.figure(figsize=(14, 10))

    # Top row: spatial images
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        img_truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax1.set_title('Ground Truth (Sharp)', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(
        img_blurry, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax2.set_title('Input (Blurry)', fontweight='bold')
    ax2.set_xlabel('Longitude')

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.imshow(
        img_reg, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax3.set_title('Regularization Only', fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.imshow(
        img_bsp, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax4.set_title('Bispectrum Constraint', fontweight='bold', color='green')
    ax4.set_xlabel('Longitude')

    # Add colorbar
    cbar_ax = fig.add_axes([0.02, 0.15, 0.01, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    # Right column: power spectrum
    ax5 = fig.add_subplot(1, 3, 3)
    ax5.plot(ps_truth, 'k-', linewidth=2, marker='o', label='Ground Truth')
    ax5.plot(ps_blurry, 'r--', linewidth=2, marker='s', label='Input (Blurry)')
    ax5.plot(ps_reg, 'b-', linewidth=1.5, marker='^', label='Reg. Only')
    ax5.plot(ps_bsp, 'g-', linewidth=2, marker='d', label='Bispectrum')
    ax5.set_yscale('log')
    ax5.set_xlabel('Degree l')
    ax5.set_ylabel('Power')
    ax5.set_title('Power Spectrum')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        'High-Frequency Recovery via Bispectrum Constraint', fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    plt.savefig('deblur_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved plot to deblur_results.png')


if __name__ == '__main__':
    results = run_deblurring_demo()
    analyze_results(*results)
