| Model (rotated test, %) | 100 | 500 | 2.5K | 12.5K | full |
|---|---|---|---|---|---|
| CNN | 17.1 ± 1.4 | 29.4 ± 0.8 | 47.0 ± 0.9 | 57.3 ± 1.1 | 78.1 ± 1.2 |
| Power spectrum | 45.5 ± 2.2 | 60.6 ± 1.0 | 69.7 ± 0.5 | 73.7 ± 0.3 | 77.7 ± 0.2 |
| S2CNN + BN | 76.1 ± 2.2 | 89.5 ± 0.8 | 93.2 ± 0.3 | 95.3 ± 0.1 | 96.1 ± 0.3 |
| Tensor product | 45.8 ± 1.3 | 72.6 ± 0.9 | 87.6 ± 0.5 | 93.9 ± 0.1 | 96.1 ± 0.1 |
| Bispectrum (ours) | 63.1 ± 2.2 | 80.5 ± 1.4 | 89.8 ± 0.5 | 94.0 ± 0.1 | 95.8 ± 0.1 |
| S2CNN + BN, R-trained | 75.8 ± 3.0 | 89.2 ± 1.1 | 93.2 ± 0.2 | 95.4 ± 0.2 | 96.6 ± 0.1 |
| Bispectrum, default settings | 62.7 ± 1.7 | 81.1 ± 1.3 | 89.4 ± 0.4 | 93.8 ± 0.2 | 95.6 ± 0.2 |

| Ours minus | n | Δ (pp) | Welch p | Holm p |
|---|---|---:|---:|---:|
| S2CNN + BN | 100 | -12.99 | 1.5e-05 | 8.9e-05 |
| S2CNN + BN | 500 | -8.98 | 1.2e-05 | 8.6e-05 |
| S2CNN + BN | 2.5K | -3.42 | 4.7e-06 | 3.8e-05 |
| S2CNN + BN | 12.5K | -1.32 | 1.2e-07 | 1.2e-06 |
| S2CNN + BN | full | -0.23 | 0.14 | 0.28 |
| Tensor product | 100 | +17.28 | 3.1e-06 | 2.8e-05 |
| Tensor product | 500 | +7.92 | 1.9e-05 | 9.7e-05 |
| Tensor product | 2.5K | +2.23 | 7.8e-05 | 0.00031 |
| Tensor product | 12.5K | +0.04 | 0.67 | 0.67 |
| Tensor product | full | -0.25 | 0.0059 | 0.018 |

Holm over the 10 tests above. Configs selected at full data and reused at every n; CNN R-trained, all other rows NR-trained unless marked.
