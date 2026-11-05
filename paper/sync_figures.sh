#!/usr/bin/env bash
# Copy the generated figures and tables into paper/ so the paper builds on its own (Overleaf).
# Captions are inlined in bispectrum.tex and are not refreshed by this script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PAPER="$REPO_ROOT/paper/experiments"
RECON_RESULTS="$REPO_ROOT/results/experiments/tooling/results/reconstruction/smnist_l15_marching/results.json"

# source (relative to the repo root) -> destination (relative to paper/experiments)
ASSETS=(
  "experiments/grid_figure/grid_contact_sheet.pdf:grid_figure/grid_contact_sheet.pdf"
  "experiments/smnist_figure/smnist_figure.pdf:smnist_figure/smnist_figure.pdf"
  "experiments/smnist_figure/smnist_protocol_table.tex:smnist_figure/smnist_protocol_table.tex"
  "benchmarks/figures/paper_fig1.pdf:benchmarks/paper_fig1.pdf"
  "benchmarks/figures/paper_fig2.pdf:benchmarks/paper_fig2.pdf"
  "benchmarks/figures/paper_fig3.pdf:benchmarks/paper_fig3.pdf"
)

for asset in "${ASSETS[@]}"; do
  src="$REPO_ROOT/${asset%%:*}"
  dst="$PAPER/${asset#*:}"
  if [[ ! -f "$src" ]]; then
    echo "missing: ${asset%%:*}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "copied ${asset%%:*}"
done

if [[ ! -f "$RECON_RESULTS" ]]; then
  echo "missing: ${RECON_RESULTS#"$REPO_ROOT"/}" >&2
  exit 1
fi
recon_dir="$PAPER/spherical_mnist_reconstruction/figures"
"$PYTHON" "$REPO_ROOT/experiments/spherical_mnist_reconstruction/degree_marching.py" plot \
  --results "$RECON_RESULTS" \
  --paper_figure_path "$recon_dir/paper_orbits.pdf"
rm -f "$recon_dir/paper_orbits.png"

echo "Captions are inlined in paper/bispectrum.tex: if a generator rewrote one" \
  "(grid_figure/caption.tex, benchmarks/figures/caption_*.tex), port the change by hand."
