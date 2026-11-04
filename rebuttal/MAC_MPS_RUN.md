# Apple-silicon (MPS) device timings

On the Mac, from a checkout of the camera-ready branch:

```bash
bash rebuttal/run_device_bench_mac.sh
```

The script sets up `.venv` (with `uv` when available), builds `torch-harmonics`
from source (no macOS wheels on PyPI), checks that bispectrum is >= 0.4.1 and MPS
is available, and writes `rebuttal/results/device_bench/m5pro_mps.json` and
`m5pro_host_cpu_t{1,8}.json`. `TAG`, `BENCH_THREADS`, `VENV` and `TH_VERSION`
override the defaults.

MPS rows cast each module to float32 (MPS has no float64); see
`environment.module_cast_dtype` in the JSON. Ship the JSON back with
`git add -f` + push, then regenerate `combined.json` / `combined.md` as described
in `EXPERIMENTS.md` (E1).
