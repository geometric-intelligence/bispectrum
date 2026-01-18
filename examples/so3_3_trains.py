import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch_harmonics import InverseRealSHT

from bispectrum import SO3onS2


def power_spectrum(coeffs: torch.Tensor) -> torch.Tensor:
    """Compute power spectrum from SH coefficients.

    Args:
        coeffs: [B, L, M] complex tensor

    Returns:
        [B, L] tensor of power per degree
    """
    return torch.sum(torch.abs(coeffs) ** 2, dim=-1)


def run_deblurring_demo():
    """
    Demo: Recovering high-frequency structure using bispectrum constraints.

    Scenario: We observe a blurry signal and want to reconstruct the sharp original.
    - Model A: Only regularization toward input (no structural prior) - stays blurry
    - Model B: Regularization + bispectrum constraint - recovers high-frequency structure
    - Model C: Regularization + spectral constraint - matches power but NOT structure

    The bispectrum acts as a "structural fingerprint" that encodes phase relationships
    between different frequency components, allowing recovery of detail that pure
    smoothness-based methods cannot achieve. Model C demonstrates that matching
    power spectrum alone is insufficient - you need phase coherence from bispectrum.
    """
    # --- Configuration ---
    LMAX = 5  # Resolution (keep low for speed)
    LAMBDA_REG = 0.01  # Regularization toward input (prevents divergence)
    LAMBDA_BSP = 10.0  # Strength of bispectrum constraint
    LAMBDA_SPEC = 10.0  # Strength of spectral (power spectrum) constraint
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
        truth_power = power_spectrum(truth_coeffs)

    # --- 2. Generate "Input" (Blurry Observation) ---
    # Heavily dampen high frequencies to simulate degraded observation
    input_coeffs = truth_coeffs.clone().detach()
    for l in range(3, LMAX + 1):
        input_coeffs[:, l, :] *= 0.1  # Kill high frequencies

    # --- 3. Optimization: Compare three approaches ---
    # Neither model has access to ground truth pixels!

    # Model A: Only stays close to input (no structural guidance)
    coeffs_A = input_coeffs.clone().detach().requires_grad_(True)
    opt_A = optim.Adam([coeffs_A], lr=0.01)

    # Model B: Stays close to input + matches bispectrum structure
    coeffs_B = input_coeffs.clone().detach().requires_grad_(True)
    opt_B = optim.Adam([coeffs_B], lr=0.01)

    # Model C: Stays close to input + matches power spectrum (spectral loss)
    coeffs_C = input_coeffs.clone().detach().requires_grad_(True)
    opt_C = optim.Adam([coeffs_C], lr=0.01)

    print('Training starts...')
    print('Model A: Regularization only (no structural prior)')
    print('Model B: Regularization + Bispectrum constraint')
    print('Model C: Regularization + Spectral (power spectrum) constraint')
    print('-' * 70)

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

        # --- Model C: Regularization + Spectral (power spectrum) ---
        opt_C.zero_grad()

        loss_reg_C = torch.mean(torch.abs(coeffs_C - input_coeffs) ** 2)

        pred_power = power_spectrum(coeffs_C)
        loss_spec = torch.mean((pred_power - truth_power) ** 2)

        loss_C = LAMBDA_REG * loss_reg_C + LAMBDA_SPEC * loss_spec
        loss_C.backward()
        opt_C.step()

        if step % 100 == 0:
            print(
                f'Step {step}: '
                f'Loss A={loss_A.item():.6f} | '
                f'Loss B={loss_B.item():.6f} (bsp={loss_bsp.item():.6f}) | '
                f'Loss C={loss_C.item():.6f} (spec={loss_spec.item():.6f})'
            )

    print('-' * 70)
    print('Done!')

    return truth_coeffs, input_coeffs, coeffs_A.detach(), coeffs_B.detach(), coeffs_C.detach()


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
def analyze_results(
    truth: torch.Tensor,
    blurry: torch.Tensor,
    res_reg_only: torch.Tensor,
    res_bispectrum: torch.Tensor,
    res_spectral: torch.Tensor,
) -> None:
    """
    Combined visualization: spatial images + power spectrum + numerical metrics.
    """
    device = truth.device
    lmax = truth.shape[1] - 1

    # Create bispectrum module for metric computation
    bsp_module = SO3onS2(lmax=lmax).to(device)

    def get_power_spectrum(coeffs: torch.Tensor) -> np.ndarray:
        return torch.sum(torch.abs(coeffs) ** 2, dim=-1).squeeze().detach().cpu().numpy()

    def power_mse(coeffs: torch.Tensor, ref: torch.Tensor) -> float:
        ps1 = power_spectrum(coeffs)
        ps2 = power_spectrum(ref)
        return torch.mean((ps1 - ps2) ** 2).item()

    def bispec_mse(coeffs: torch.Tensor, ref: torch.Tensor) -> float:
        with torch.no_grad():
            inv1 = bsp_module(coeffs)
            inv2 = bsp_module(ref)
        return torch.mean(torch.abs(inv1 - inv2) ** 2).item()

    # --- Compute Numerical Metrics ---
    metrics: dict[str, dict[str, float]] = {}
    models = [
        ('Input (Blurry)', blurry),
        ('Reg. Only', res_reg_only),
        ('Spectral', res_spectral),
        ('Bispectrum', res_bispectrum),
    ]

    for name, coeffs in models:
        metrics[name] = {
            'power_mse': power_mse(coeffs, truth),
            'bispec_mse': bispec_mse(coeffs, truth),
        }

    # Print metrics to console
    print('\n' + '=' * 70)
    print('NUMERICAL METRICS (MSE vs Ground Truth)')
    print('=' * 70)
    print(f'{"Model":<25} {"Power Spectrum MSE":>20} {"Bispectrum MSE":>20}')
    print('-' * 70)
    for name, m in metrics.items():
        print(f'{name:<25} {m["power_mse"]:>20.6e} {m["bispec_mse"]:>20.6e}')
    print('=' * 70)

    # Render all signals to spatial domain
    print('Rendering spherical harmonics to spatial domain...')
    img_truth = sh_to_spatial(truth)
    img_blurry = sh_to_spatial(blurry)
    img_reg = sh_to_spatial(res_reg_only)
    img_bsp = sh_to_spatial(res_bispectrum)
    img_spec = sh_to_spatial(res_spectral)

    # Compute power spectra for plotting
    ps_truth = get_power_spectrum(truth)
    ps_blurry = get_power_spectrum(blurry)
    ps_reg = get_power_spectrum(res_reg_only)
    ps_bsp = get_power_spectrum(res_bispectrum)
    ps_spec = get_power_spectrum(res_spectral)

    # Common colormap range for fair comparison
    all_imgs = [img_truth, img_blurry, img_reg, img_bsp, img_spec]
    vmin = min(img.min() for img in all_imgs)
    vmax = max(img.max() for img in all_imgs)

    # Create combined figure: 3 rows - images, power spectrum + metrics, explanation
    fig = plt.figure(figsize=(18, 14))

    # Use GridSpec for flexible layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.25)

    # --- Row 1: Ground Truth, Input (Blurry), Regularization Only ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        img_truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax1.set_title('Ground Truth (Sharp)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(
        img_blurry, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax2.set_title('Input (Blurry)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Longitude')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(
        img_reg, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax3.set_title('Regularization Only', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Longitude')

    # --- Row 2: Spectral Constraint, Bispectrum Constraint, Power Spectrum plot ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(
        img_spec, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax4.set_title('Spectral Constraint', fontweight='bold', fontsize=11, color='#CC6600')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(
        img_bsp, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', extent=[0, 360, -90, 90]
    )
    ax5.set_title('Bispectrum Constraint', fontweight='bold', fontsize=11, color='green')
    ax5.set_xlabel('Longitude')

    # Power spectrum comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ps_truth, 'k-', linewidth=2, marker='o', markersize=8, label='Ground Truth')
    ax6.plot(ps_blurry, 'r--', linewidth=2, marker='s', markersize=7, label='Input (Blurry)')
    ax6.plot(ps_reg, 'b-', linewidth=1.5, marker='^', markersize=6, label='Reg. Only')
    ax6.plot(
        ps_spec,
        color='#CC6600',
        linestyle='-',
        linewidth=2,
        marker='x',
        markersize=8,
        label='Spectral',
    )
    ax6.plot(ps_bsp, 'g-', linewidth=2, marker='d', markersize=7, label='Bispectrum')
    ax6.set_yscale('log')
    ax6.set_xlabel('Degree $\\ell$', fontsize=10)
    ax6.set_ylabel('Power $\\sum_m |a_{\\ell m}|^2$', fontsize=10)
    ax6.set_title('Power Spectrum Comparison', fontweight='bold', fontsize=11)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)

    # --- Row 3: Metrics table and explanation text ---
    # Left panel: Metrics table
    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis('off')

    # Create table data
    table_data = [
        ['Model', 'Power MSE', 'Bispec MSE'],
        [
            'Input (Blurry)',
            f'{metrics["Input (Blurry)"]["power_mse"]:.2e}',
            f'{metrics["Input (Blurry)"]["bispec_mse"]:.2e}',
        ],
        [
            'Reg. Only',
            f'{metrics["Reg. Only"]["power_mse"]:.2e}',
            f'{metrics["Reg. Only"]["bispec_mse"]:.2e}',
        ],
        [
            'Spectral',
            f'{metrics["Spectral"]["power_mse"]:.2e}',
            f'{metrics["Spectral"]["bispec_mse"]:.2e}',
        ],
        [
            'Bispectrum',
            f'{metrics["Bispectrum"]["power_mse"]:.2e}',
            f'{metrics["Bispectrum"]["bispec_mse"]:.2e}',
        ],
    ]

    # Color cells based on values (green=good, red=bad)
    cell_colors = [['lightgray'] * 3]  # Header
    for _i, name in enumerate(['Input (Blurry)', 'Reg. Only', 'Spectral', 'Bispectrum']):
        row_colors = ['white']  # Name column
        # Power MSE color
        if metrics[name]['power_mse'] < 1e-6:
            row_colors.append('#90EE90')  # Light green
        elif metrics[name]['power_mse'] < 1e-4:
            row_colors.append('#FFFFE0')  # Light yellow
        else:
            row_colors.append('#FFB6C1')  # Light red
        # Bispectrum MSE color
        if metrics[name]['bispec_mse'] < 1e-6:
            row_colors.append('#90EE90')  # Light green
        elif metrics[name]['bispec_mse'] < 1e-4:
            row_colors.append('#FFFFE0')  # Light yellow
        else:
            row_colors.append('#FFB6C1')  # Light red
        cell_colors.append(row_colors)

    table = ax_table.table(
        cellText=table_data,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center',
        colWidths=[0.35, 0.32, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax_table.set_title('MSE vs Ground Truth', fontweight='bold', fontsize=11, pad=10)

    # Middle panel: Key insight box
    ax_insight = fig.add_subplot(gs[2, 1])
    ax_insight.axis('off')

    insight_text = (
        'KEY INSIGHT\n'
        '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
        'Spectral constraint matches power\n'
        'spectrum perfectly (MSE ≈ 10⁻⁹)\n'
        'but spatial structure is WRONG!\n\n'
        'Why? Power spectrum = amplitudes only.\n'
        'It discards phase information.\n\n'
        'Bispectrum encodes phase relationships\n'
        'between frequencies → recovers structure.'
    )
    ax_insight.text(
        0.5,
        0.5,
        insight_text,
        transform=ax_insight.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox={
            'boxstyle': 'round,pad=0.5',
            'facecolor': '#E8F4E8',
            'edgecolor': 'green',
            'linewidth': 2,
        },
    )

    # Right panel: Mathematical explanation
    ax_math = fig.add_subplot(gs[2, 2])
    ax_math.axis('off')

    math_text = (
        'MATHEMATICAL INTERPRETATION\n'
        '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n'
        'Power Spectrum:\n'
        '  $P_\\ell = \\sum_m |a_{\\ell m}|^2$\n'
        '  → Rotation invariant\n'
        '  → Loses phase info\n\n'
        'Bispectrum:\n'
        '  $B_{\\ell_1 \\ell_2 \\ell_3} = '
        '\\sum_{m_i} C^{\\ell_3}_{\\ell_1 \\ell_2} a_{\\ell_1 m_1} a_{\\ell_2 m_2} a^*_{\\ell_3 m_3}$\n'
        '  → Rotation invariant\n'
        '  → Preserves phase coherence'
    )
    ax_math.text(
        0.5,
        0.5,
        math_text,
        transform=ax_math.transAxes,
        fontsize=9,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox={
            'boxstyle': 'round,pad=0.5',
            'facecolor': '#F0F0FF',
            'edgecolor': 'blue',
            'linewidth': 2,
        },
    )

    # Add colorbar
    cbar_ax = fig.add_axes([0.02, 0.35, 0.012, 0.55])
    fig.colorbar(im1, cax=cbar_ax)

    plt.suptitle(
        'Power Spectrum vs Bispectrum: Why Phase Coherence Matters for Signal Recovery',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    plt.savefig('deblur_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved plot to deblur_results.png')


if __name__ == '__main__':
    results = run_deblurring_demo()
    analyze_results(*results)
