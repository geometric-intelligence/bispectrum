# Bispectral pooling head ablation

90 runs. Mean ± std over seeds (ddof=1). Primary metric: test accuracy (OrganMNIST3D) or test AUC (PCam) at the best-validation checkpoint; `final` is the same metric at the last epoch. `p(pre)` / `p(base)`: Welch t-test against the original head (`pre_log`) and against the best baseline of the setting.

## organ3d / ch16_32_full  (metric: accuracy)

| model | n | test (best val) | test (final) | best val AUC | params | min/run | p(pre) | p(base) |
|---|---|---|---|---|---|---|---|---|
| `pre_log` | 3 | 73.9 ± 1.5 | 75.2 ± 0.3 | 0.9909 ± 0.0007 | 6,338,523 | 78 | - | 0.247 |
| `post_log` | 3 | 77.9 ± 1.9 | 78.7 ± 1.6 | 0.9960 ± 0.0018 | 6,338,523 | 76 | 0.045 | 0.297 |
| `post_log_power` ★ | 3 | 78.1 ± 1.2 | 77.7 ± 0.3 | 0.9963 ± 0.0018 | 6,348,763 | 77 | 0.020 | 0.226 |
| `post_cbrt_power` | 3 | 77.5 ± 0.7 | 78.3 ± 1.2 | 0.9939 ± 0.0029 | 6,348,763 | 75 | 0.033 | 0.328 |
| `max_pool` | 3 | 76.0 ± 2.1 | 78.1 ± 0.7 | 0.9928 ± 0.0011 | 5,985,851 | 71 | 0.247 | - |
| `norm_pool` | 3 | 68.2 ± 3.7 | 69.3 ± 3.2 | 0.9872 ± 0.0006 | 5,986,171 | 89 | 0.103 | 0.048 |

Best baseline: `max_pool`. Validation winner among heads: `post_log_power` (★).

## organ3d / ch4_8_full  (metric: accuracy)

| model | n | test (best val) | test (final) | best val AUC | params | min/run | p(pre) | p(base) |
|---|---|---|---|---|---|---|---|---|
| `pre_log` | 3 | 73.6 ± 2.0 | 74.6 ± 0.8 | 0.9894 ± 0.0011 | 463,167 | 12 | - | 0.691 |
| `post_log` | 3 | 78.6 ± 0.3 | 78.5 ± 0.3 | 0.9955 ± 0.0020 | 463,167 | 12 | 0.049 | 0.001 |
| `post_log_power` | 3 | 76.9 ± 2.4 | 77.7 ± 1.3 | 0.9941 ± 0.0009 | 465,727 | 11 | 0.140 | 0.098 |
| `post_cbrt_power` ★ | 3 | 79.0 ± 3.7 | 78.7 ± 3.6 | 0.9955 ± 0.0025 | 465,727 | 11 | 0.113 | 0.106 |
| `max_pool` | 3 | 73.1 ± 0.6 | 73.4 ± 1.0 | 0.9890 ± 0.0009 | 374,423 | 8 | 0.691 | - |
| `norm_pool` | 3 | 63.3 ± 2.7 | 63.7 ± 2.1 | 0.9780 ± 0.0018 | 374,503 | 11 | 0.007 | 0.020 |

Best baseline: `max_pool`. Validation winner among heads: `post_cbrt_power` (★).

## organ3d / ch4_8_n100  (metric: accuracy)

| model | n | test (best val) | test (final) | best val AUC | params | min/run | p(pre) | p(base) |
|---|---|---|---|---|---|---|---|---|
| `pre_log` | 3 | 61.0 ± 0.9 | 60.9 ± 1.0 | 0.9787 ± 0.0039 | 463,167 | 12 | - | 0.169 |
| `post_log` | 3 | 61.1 ± 2.6 | 61.2 ± 2.9 | 0.9848 ± 0.0049 | 463,167 | 12 | 0.926 | 0.359 |
| `post_log_power` | 3 | 62.4 ± 2.7 | 63.2 ± 2.2 | 0.9850 ± 0.0070 | 465,727 | 12 | 0.465 | 0.746 |
| `post_cbrt_power` ★ | 3 | 62.7 ± 2.8 | 62.8 ± 2.3 | 0.9858 ± 0.0039 | 465,727 | 12 | 0.410 | 0.853 |
| `max_pool` | 3 | 63.1 ± 1.8 | 63.3 ± 1.3 | 0.9795 ± 0.0013 | 374,423 | 10 | 0.169 | - |
| `norm_pool` | 3 | 57.9 ± 3.0 | 59.1 ± 2.3 | 0.9770 ± 0.0044 | 374,503 | 13 | 0.208 | 0.075 |

Best baseline: `max_pool`. Validation winner among heads: `post_cbrt_power` (★).

## pcam / n100  (metric: auc)

| model | n | test (best val) | AUC exact rot | AUC 45° rot | test (final) | best val AUC | params | min/run | p(pre) | p(base) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pre_log` | 3 | 0.8116 ± 0.0241 | 0.8119 ± 0.0243 | 0.7749 ± 0.0708 | 0.8152 ± 0.0060 | 0.8400 ± 0.0169 | 128,345 | 48 | - | 0.976 |
| `post_log` ★ | 3 | 0.8236 ± 0.0207 | 0.8240 ± 0.0208 | 0.7856 ± 0.0621 | 0.8149 ± 0.0224 | 0.8538 ± 0.0069 | 128,345 | 52 | 0.547 | 0.708 |
| `post_log_power` | 3 | 0.8132 ± 0.0306 | 0.8135 ± 0.0306 | 0.7536 ± 0.0453 | 0.8149 ± 0.0267 | 0.8500 ± 0.0026 | 134,125 | 55 | 0.944 | 0.981 |
| `post_cbrt_power` | 3 | 0.8212 ± 0.0029 | 0.8212 ± 0.0027 | 0.7885 ± 0.0466 | 0.8072 ± 0.0169 | 0.8468 ± 0.0125 | 134,125 | 50 | 0.561 | 0.753 |
| `norm_pool` | 3 | 0.8022 ± 0.0381 | 0.8025 ± 0.0377 | 0.7869 ± 0.0120 | 0.8000 ± 0.0310 | 0.8289 ± 0.0143 | 109,815 | 47 | 0.740 | 0.769 |
| `standard` | 3 | 0.8125 ± 0.0419 | 0.8149 ± 0.0407 | 0.8167 ± 0.0420 | 0.8026 ± 0.0290 | 0.8498 ± 0.0034 | 101,959 | 17 | 0.976 | - |

Best baseline: `standard`. Validation winner among heads: `post_log` (★).

## pcam / n12500  (metric: auc)

| model | n | test (best val) | AUC exact rot | AUC 45° rot | test (final) | best val AUC | params | min/run | p(pre) | p(base) |
|---|---|---|---|---|---|---|---|---|---|---|
| `pre_log` | 3 | 0.9392 ± 0.0113 | 0.9393 ± 0.0114 | 0.9162 ± 0.0187 | 0.9419 ± 0.0106 | 0.9458 ± 0.0025 | 128,345 | 63 | - | 0.534 |
| `post_log` ★ | 3 | 0.9363 ± 0.0065 | 0.9364 ± 0.0064 | 0.8931 ± 0.0434 | 0.9382 ± 0.0084 | 0.9570 ± 0.0035 | 128,345 | 58 | 0.730 | 0.680 |
| `post_log_power` | 3 | 0.9400 ± 0.0082 | 0.9401 ± 0.0081 | 0.9146 ± 0.0089 | 0.9342 ± 0.0028 | 0.9520 ± 0.0006 | 134,125 | 69 | 0.921 | 0.424 |
| `post_cbrt_power` | 3 | 0.9359 ± 0.0050 | 0.9358 ± 0.0051 | 0.9112 ± 0.0101 | 0.9310 ± 0.0032 | 0.9526 ± 0.0031 | 134,125 | 67 | 0.676 | 0.713 |
| `norm_pool` | 3 | 0.9331 ± 0.0106 | 0.9332 ± 0.0105 | 0.9108 ± 0.0036 | 0.9313 ± 0.0121 | 0.9512 ± 0.0038 | 109,815 | 54 | 0.534 | - |
| `standard` | 3 | 0.9283 ± 0.0094 | 0.9280 ± 0.0102 | 0.9221 ± 0.0139 | 0.9238 ± 0.0135 | 0.9499 ± 0.0040 | 101,959 | 20 | 0.271 | 0.589 |

Best baseline: `norm_pool`. Validation winner among heads: `post_log` (★).

## Head decision (best-validation AUC, no test metrics involved)

- `post_log`: mean rank 1.80
- `post_cbrt_power`: mean rank 2.00
- `post_log_power`: mean rank 2.20
- `pre_log`: mean rank 4.00

**Decision: `post_log`**
