| Model | Params | NR/NR | R/R | NR/R | Latency (ms) | p vs. ours (NR/NR, R/R, NR/R) |
|---|---:|---|---|---|---:|---|
| CNN (equirectangular) | 185K | 99.1 ± 0.1 | 78.1 ± 1.2 | 13.4 ± 0.2 | 0.20 | 9e-10, 3.4e-06, 6.9e-16 |
| Power spectrum + MLP | 11K | 77.8 ± 0.2 | 77.8 ± 0.2 | 77.7 ± 0.2 | 1.68 | 3.1e-14, 6.5e-16, 1.5e-14 |
| S2CNN, published (Cohen et al., 2018) | --- | 96 | 95 | 94 | --- | --- |
| S2CNN + BatchNorm (matched) | 155K | 97.1 ± 0.1 | 96.6 ± 0.1 | 96.1 ± 0.3 | 1.02 | 1.7e-07, 2.5e-05, 0.14 |
| Tensor-product net (matched) | 386K | 96.1 ± 0.1 | 96.1 ± 0.1 | 96.1 ± 0.1 | 93.7 | 0.0036, 0.0046, 0.0059 |
| Bispectrum + MLP (ours) | 230K | 95.8 ± 0.1 | 95.7 ± 0.1 | 95.8 ± 0.1 | 0.26 | --- |

Spherical MNIST, test accuracy (%) under the protocols of Cohen et al. (2018),
mean +/- sample std over 5 seeds, and end-to-end forward latency (spherical grid
in, logits out; float32, batch 16, H100, median). Matched baselines are our runs
in the same training pipeline; the published S2CNN row is transcribed from Cohen
et al. (2018). Bispectrum, S2CNN + BatchNorm and the tensor-product net were each
selected on validation accuracy from a batch-size x learning-rate grid (the
tensor-product grid also covers channels and internal band-limit); the CNN and
power spectrum use the default settings (batch 256, lr 1e-3). S2CNN with the
published architecture and training settings (no BatchNorm) reaches
81.2 / 80.5 / 80.7 (NR/NR / R/R / NR/R) in our pipeline.
