#!/usr/bin/env bash
# Apple-silicon device timings (MPS + CPU at pinned thread counts) for the
# camera-ready portability matrix. Run from a checkout of this branch on the Mac:
#
#   bash rebuttal/run_device_bench_mac.sh
#
# Writes rebuttal/results/device_bench/${TAG}_mps.json and ${TAG}_host_cpu_t<n>.json.
# Environment: TAG (default m5pro), BENCH_THREADS (default "1 8"), VENV (default .venv).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-m5pro}"
BENCH_THREADS="${BENCH_THREADS:-1 8}"
VENV="${VENV:-$REPO_ROOT/.venv}"
OUT="$REPO_ROOT/rebuttal/results/device_bench"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run this on the Mac; Linux hosts use: bash rebuttal/run_camera_ready.sh timings <tag>" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  [[ -x "$VENV/bin/python" ]] || uv venv --python 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install -q -e .
else
  [[ -x "$VENV/bin/python" ]] || python3.12 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q -e .
fi
PY="$VENV/bin/python"

"$PY" - <<'PY'
import sys

import torch

import bispectrum

print('bispectrum', bispectrum.__version__, 'from', bispectrum.__file__)
print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())
if tuple(int(part) for part in bispectrum.__version__.split('.')[:3]) < (0, 4, 0):
    sys.exit('bispectrum < 0.4.0 in this venv')
if not torch.backends.mps.is_available():
    sys.exit('MPS unavailable')
PY

mkdir -p "$OUT"
"$PY" rebuttal/benchmark_devices.py --device mps --output "$OUT/${TAG}_mps.json" | tee "$OUT/${TAG}_mps.log"
for threads in $BENCH_THREADS; do
  "$PY" rebuttal/benchmark_devices.py --device cpu --threads "$threads" \
    --output "$OUT/${TAG}_host_cpu_t${threads}.json" | tee "$OUT/${TAG}_host_cpu_t${threads}.log"
done

echo
echo "Done. Ship the JSON back with either:"
echo "  git add -f rebuttal/results/device_bench/${TAG}_*.json && git commit -m 'device timings: ${TAG}' && git push"
echo "  scp rebuttal/results/device_bench/${TAG}_*.json <h100>:bispectrum/rebuttal/results/device_bench/"
