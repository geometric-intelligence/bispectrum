# Camera-ready rerun vs. the paper (arXiv 2605.07270)

"Old" is the number printed in `paper/bispectrum.tex`; "new" is the camera-ready rerun with the
fixed bispectral head (`BispectrumHead(nonlin='post', power=False, scale='log')`), bispectrum
0.4.0, a fixed epoch budget ending on an SGDR cycle (no early stopping), and one batch size per
dataset. New numbers come from the `*.txt` files next to this one (mean ± population std over
seeds, as in the paper); significance tests are in `significance.md` (rotated test) and
`significance_canonical.md`.

**Runs.** 523 PCam/OrganMNIST3D jobs (job runner, 3 hosts) + 160 Spherical MNIST runs (H100
pipeline). All 523 jobs have done markers; 0 stale runs. The live result trees record 9 commits
(`c215e2b`, `6f4dd3f`, `a814e56`, `9f8dda8`, `542a1e6`, `b98c532`, `ee9f1b8`, `6550993`,
`40a650f`); `git diff` between them is empty for `src/`, `experiments/**/{train,model,data}*.py`,
the sweep scripts, `pyproject.toml` and `uv.lock` — they differ only in the job runner and the
manifest.

**Protocol changes vs. the paper.**

| | paper | camera-ready |
|---|---|---|
| PCam sweep / matched table | 10% data (26,214), early stopping (patience 10, ≤50 epochs), 1 seed (sweep) / 3 seeds (table) | n = 12,500, 30 epochs (10 on the full 262,144), 3 seeds everywhere |
| PCam data efficiency | 1% … 10%, 3 seeds | n ∈ {100, 500, 2,500} × 5 seeds; 12,500 and full (262,144) × 3 seeds |
| OrganMNIST3D | 100 epochs, 3 seeds | 70 epochs, 10 seeds; n ∈ {50, 100, 250, 500, 971} |
| Spherical MNIST | 3 seeds | 5 seeds, 30 epochs |

⚠ marks a claim in the paper text that no longer holds.

## Table `tab:smnist-results` — Spherical MNIST (`smnist.txt`, capacity rows from the capacity sweep)

| Model | Params | NR/NR old → new | R/R old → new | NR/R old → new | σ_rot old → new |
|---|---|---|---|---|---|
| Standard CNN | 185K | 0.991±0.000 → 0.991±0.000 | 0.781±0.003 → 0.781±0.010 | 0.138±0.003 → 0.134±0.002 | 0.044 → 0.042 |
| Power spec. + MLP | 11K | 0.780±0.002 → 0.778±0.001 | 0.780±0.002 → 0.778±0.001 | 0.777±0.001 → 0.777±0.002 | 3e-4 → 3.5e-4 |
| Power spec. (matched) | 166K → 144K (h=1024) | 0.794±0.011 → 0.798±0.001 | 0.794±0.011 → not run (capacity sweep is C-trained) | 0.792±0.010 → 0.795±0.002 | 3e-4 → 2.5e-4 |
| **Bispectrum + MLP** | 232K → 230K | 0.950±0.000 → **0.956±0.002** | 0.949±0.002 → **0.955±0.001** | 0.951±0.001 → **0.956±0.002** | 1e-4 → 1.3e-4 |

Text claims:
- "bispectrum 95.0/94.9/95.1 is 1–2 pp lower than Cohen's 96/95/94" ⚠ → now 95.6/95.5/95.6:
  −0.4 pp, +0.5 pp, **+1.6 pp** vs. Cohen. Bispectrum now beats the published S2CNN on NR/R.
- "power spectrum ~17 pp deficit" → 17.8 pp (C/C) and 17.9 pp (C/R). Holds.
- "scaling the power-spectrum MLP 11K → 166K yields only +1.4 pp (NR/R)" → +1.8 pp to 144K,
  +2.4 pp to 550K (C/R 0.801). Holds qualitatively; bispectrum at 25K (h=32) already reaches
  0.940.
- "standard CNN collapses 99.1% → 13.8%" → 99.1% → 13.4%. Holds.

## Table `tab:pcam-results` — PCam at ~100K params (canonical test)

Old: 10% data. New: n = 12,500 growth sweep, the growth rate closest to the paper's parameter count.

| Model | Params old → new | AUC old → new | ACC old → new |
|---|---|---|---|
| Standard (augmented) | 102K → 102K (gr12) | .896±.010 → .928±.008 | .826±.012 → .854±.011 |
| Gated | 136K → 136K (gr3) | .941±.009 → .921±.004 | .870±.004 → .834±.007 |
| **Bispectrum (CnonCn)** | 128K → 128K (gr4) | .941±.004 → **.946±.009** | .872±.006 → **.873±.012** |
| Bispectrum (smallest) | — → 80K (gr3) | — → **.951±.014** | — → **.880±.022** |
| Fourier-ELU | 110K → 110K (gr4) | .945±.004 → .922±.021 | .855±.006 → .842±.028 |
| NormReLU | 110K → 110K (gr4) | .942±.004 → .913±.011 | .858±.007 → .826±.010 |
| Norm pool (new model) | — → 110K (gr4) | — → .935±.012 | — → .847±.016 |

Text claims:
- "all four equivariant methods substantially outperform the augmented CNN (0.896)" ⚠ → the
  augmented CNN is now 0.928, above NormReLU (0.913), gated (0.921) and Fourier-ELU (0.922);
  only the bispectrum (0.946 / 0.951) and norm pool (0.935) beat it.
- "Fourier-ELU leads slightly (0.945)" ⚠ → the bispectrum leads; Fourier-ELU is 4th.
- "At 1% data the bispectrum leads (0.912) vs NormReLU 0.873, Fourier-ELU 0.876, gated 0.884" →
  n = 2,500 (0.95%): bispectrum **0.931±0.012**, gated 0.924, norm pool 0.914, aug. CNN 0.912,
  Fourier-ELU 0.911, NormReLU 0.903. Still leads, by 0.7 pp instead of 2.8 pp. Rotated-test
  Welch p: vs NormReLU < 0.001, vs Fourier-ELU 0.011, others n.s.
- "bispectrum improves with scale (+0.9, 0.943 at 920K) while standard/norm/Fourier-ELU degrade
  (−1.8/−2.6/−0.9)" ⚠ → reversed: bispectrum 0.951 (80K) → 0.933 (920K) = −1.7; augmented CNN
  +1.3 (0.916 → 0.928); NormReLU −1.4; Fourier-ELU −1.8.
- "bispectrum AUC varies by only ~1 pp across 80K–920K; norm swings >7 pp" ⚠ → bispectrum range
  2.8 pp (0.923–0.951), NormReLU 1.5 pp (0.898–0.913).
- "bispectrum and Fourier-ELU have the lowest cross-seed variance (±0.004)" ⚠ → at ~100K gated is
  lowest (±0.004); bispectrum ±0.009–0.014, Fourier-ELU ±0.021.
- Full data (new cell, rotated AUC, `significance.md`): bispectrum 0.928 vs gated 0.944,
  NormReLU 0.936, Fourier-ELU 0.931 — no significant difference, bispectrum not ahead.

Four low-data PCam runs have a rotated AUC 0.15–0.19 below their canonical AUC (gated n=100
seed 42; NormReLU n=500 seed 101112; norm pool n=500 seed 123) or are weak on both (norm pool
n=100 seed 789). All exited 0 with the full epoch budget; they produce the wide error bars at
n ≤ 500. The grid caption's "canonical and rotated differ by at most 0.018" is computed on
full-training-set cells only.

## Table `tab:organ3d-results` — OrganMNIST3D, channels 4→8, full data (`organ3d.txt`)

| Model | Params | ACC old → new | AUC old → new | Rot ACC old → new | σ_rot old → new |
|---|---|---|---|---|---|
| Standard 3D CNN (aug.) | 16K | .601±.017 → .626±.044 | .951±.002 → .956±.008 | .598 → .624 | .0115 → .0139 |
| O-equiv + norm pool | 375K | .568±.099 → .629±.024 | .940±.021 → .952±.005 | .568 → .629 | .0000 → .0000 |
| O-equiv + max pool | 374K | .730±.033 → .736±.022 | .972±.005 → .973±.003 | .730 → .736 | .0000 → .0000 |
| **O-equiv + bispectrum** | 463K | .726±.027 → **.785±.006** | .972±.004 → **.980±.002** | .726 → **.785** | .0005 → .0002 (max over seeds .0008) |

Text claims:
- "bispectrum (72.6%) matches max pool (73.0%)" → bispectrum **+5.0 pp** over max pool
  (78.5 vs 73.6, p < 0.001, 10 seeds) with a 3.5× smaller seed spread.
- "augmented CNN drops ~2.5 pp under rotation" ⚠ → 62.6 → 62.4 (−0.2 pp mean over the 24
  rotations; σ_rot 1.4 pp).
- "all equivariant models σ_rot ≤ 0.0005" → mean 0.0002; one bispectrum seed reaches 0.0008.
- "at 5% data the bispectrum reaches 33.2% vs 18.9% for max pool; the advantage widens at reduced
  data" ⚠ → n = 50 (5.1%): 53.9±5.9 vs 52.3±2.7 (+1.6 pp, p = 0.45). The gap is now largest at
  n = 500 (+6.7 pp, p < 0.001) and full data (+5.0 pp); at n ≤ 250 it is not significant.
- "at higher capacity (6M) max pool overtakes the bispectrum" ⚠ → it does not: 16→32 bispectrum
  77.6 vs max pool 76.0 (+1.6 pp, p = 0.096); 8→16 +2.0 pp (p = 0.029).
- New: bispectrum vs augmented CNN and norm pool is significant at every n ≥ 100 (`significance.md`).

| Width | max pool | bispectrum | norm pool | aug. CNN |
|---|---|---|---|---|
| 4→8 | .736±.022 (374K) | .785±.006 (463K) | .629±.024 | .626±.044 (16K) |
| 8→16 | .756±.015 (1.5M) | .776±.020 (1.7M) | .662±.032 | .692±.026 (63K) |
| 16→32 | .760±.014 (6.0M) | .776±.024 (6.3M) | .681±.033 | .710±.020 (251K) |

## Figures

| Paper figure | New file | Change |
|---|---|---|
| `fig:pcam-pareto` (`experiments/pcam/pareto_analysis.pdf`) | `pareto_analysis.{png,pdf}`, `data_pareto_analysis.png` | 3 seeds per point instead of 1; scaling claims above flip |
| `fig:data-efficiency` (`experiments/organ3d/data_efficiency.pdf`) | `data_efficiency.{png,pdf}` | n = 50…971 with 10 seeds, all four models |
| `fig:organ3d-rotation` (`experiments/organ3d/organ3d_analysis.pdf`) | `organ3d_analysis.{png,pdf}` | augmented CNN now included (R-trained) |
| `fig:smnist-analysis` (`experiments/spherical_mnist/smnist_analysis.pdf`) | `smnist_analysis.{png,pdf}`, `smnist_data_efficiency.{png,pdf}` | 5 seeds |
| grid figure (`experiments/grid_figure/`) | not committed yet | `make_grid_figure.py` is being revised in another session; regenerate with `run_camera_ready.sh analyze` once it lands |

Seed-level significance (`significance.md`, rotated test, Welch p < 0.05 uncorrected): the
bispectrum is significantly better in 16 of 21 OrganMNIST3D comparisons, 10 of 10 Spherical MNIST
and 3 of 25 PCam, and significantly worse in none.

Not rerun (no training; library unchanged): reconstruction, device timings. The S2CNN and e3nn
tensor-product baselines for Spherical MNIST are running on `johmathe/camera-ready-baselines`.
