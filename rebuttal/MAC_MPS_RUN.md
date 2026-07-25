# E1 Apple-silicon (MPS) benchmark — local run instructions

Goal: produce `mac_mps.json` / `mac_cpu.json` so the rebuttal's portability
matrix covers Apple silicon (answers Reviewer yWuC Q4; can be posted as a
follow-up comment if it lands before the discussion deadline).

## 1. Setup (on the Mac)

```bash
git clone git@github.com:geometric-intelligence/bispectrum.git  # or git pull
cd bispectrum
git checkout neurips-2026-rebuttal   # must include the tagger/train-metric fix commits
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.10,<2.12" "numpy>=1.22.4"
python -m pip install --no-build-isolation \
  "https://github.com/NVIDIA/torch-harmonics/archive/refs/tags/v0.9.1.tar.gz"
python -m pip install --no-deps -e .
python -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available())"
```

`torch-harmonics>=0.9` has no macOS wheel, so the normal extras install cannot
resolve on Apple silicon. The tagged source build above provides its CPU/MPS
implementation without requiring CUDA.

## 2. Run the benchmarks

```bash
mkdir -p rebuttal/results/device_bench
python rebuttal/benchmark_devices.py --device mps \
  --output rebuttal/results/device_bench/mac_mps.json \
  | tee rebuttal/results/device_bench/mac_mps.log
python rebuttal/benchmark_devices.py --device cpu \
  --output rebuttal/results/device_bench/mac_cpu.json \
  | tee rebuttal/results/device_bench/mac_cpu.log
```

Notes:

- Expected runtime: roughly 5–20 minutes per device; first-time CG/Bessel
  cache construction can dominate.
- Unsupported operations, if any, are retained as **explicit errors** in the
  JSON rather than silently dropped.
- The environment (platform, PyTorch/backend version, dtype, batch 16) is
  recorded inside each JSON automatically.

## 3. Copy results back to the GPU machine

```bash
scp rebuttal/results/device_bench/mac_*.json rebuttal/results/device_bench/mac_*.log \
  bongo:/home/johmathe/bispectrum/rebuttal/results/device_bench/
```

## 4. Refresh the combined analysis (on bongo)

```bash
.venv/bin/python rebuttal/analyze_device_benchmarks.py \
  rebuttal/results/device_bench/a100_cuda.json \
  rebuttal/results/device_bench/mac_mps.json \
  rebuttal/results/device_bench/a100_host_cpu.json \
  rebuttal/results/device_bench/mac_cpu.json \
  --output rebuttal/results/device_bench/combined.json \
  | tee rebuttal/results/device_bench/combined.md
```

Then update the portability table in `rebuttal/reviewer_yWuC.md` and
`rebuttal/rebuttal.tex` (the yWuC response has ~1.8k characters of headroom
under the 10,000-character limit for the added row).
