"""Move stale experiment results out of the sweep output directories.

Every sweep script skips a run whose ``results.json`` already exists, so results
written before the per-epoch sample floor, before bispectrum 0.4.0, or with
patience-based early stopping must be moved aside before the camera-ready
reruns. Grid run directories are classified with
``make_grid_figure._stale_reason`` so exactly the runs the grid figure drops
are archived. Rebuttal result trees without grid provenance fields
(reconstruction, device timings) are archived whole, once.

    python rebuttal/archive_stale_results.py --dry_run
    python rebuttal/archive_stale_results.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'experiments'))

from make_grid_figure import _stale_reason  # noqa: E402

ARCHIVE_TAG: Final = {
    'pre-floor': 'prefloor',
    'pre-0.4.0': 'pre040',
    'early-stopped': 'earlystop',
    'pre-head-fix': 'preheadfix',
}

# Grid result roots: individual run dirs are moved to <root>_<tag>/<relative path>.
GRID_ROOTS: Final = (
    'experiments/pcam/pcam_results_data_pareto',
    'experiments/pcam/pcam_results_pareto',
    'experiments/organ3d/organ3d_results',
    'experiments/organ3d/organ3d_results_bs16_control',
    'experiments/spherical_mnist/smnist_results',
    'experiments/spherical_mnist/smnist_results_capacity',
    'rebuttal/results/smnist_ablation',
)

# Whole trees whose results.json carry no grid provenance; all predate 0.4.0.
WHOLE_DIRS: Final = (
    'rebuttal/results/reconstruction',
    'rebuttal/results/reconstruction_smoke',
    'rebuttal/results/device_bench',
)


def _move(src: Path, dst: Path, dry_run: bool) -> None:
    print(f'{"would move" if dry_run else "move"}: {src} -> {dst}')
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def archive_grid_root(root: Path, dry_run: bool) -> Counter[str]:
    """Move every stale run directory under *root* into ``<root>_<tag>/``."""
    moved: Counter[str] = Counter()
    if not root.is_dir():
        return moved
    for results_path in sorted(root.rglob('results.json')):
        try:
            run = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            print(f'skip (unreadable): {results_path}')
            continue
        reason = _stale_reason(run)
        if reason is None:
            continue
        run_dir = results_path.parent
        archive_root = root.with_name(f'{root.name}_{ARCHIVE_TAG[reason]}')
        _move(run_dir, archive_root / run_dir.relative_to(root), dry_run)
        moved[reason] += 1
    return moved


def archive_whole_dir(src: Path, dry_run: bool) -> bool:
    """Move *src* to ``<src>_pre040`` unless that archive already exists."""
    if not src.is_dir():
        return False
    dst = src.with_name(f'{src.name}_pre040')
    if dst.exists():
        print(f'keep (archive exists): {src}')
        return False
    _move(src, dst, dry_run)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--repo_root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--dry_run', action='store_true', help='Print the moves without performing them.'
    )
    args = parser.parse_args()

    total: Counter[str] = Counter()
    for rel in GRID_ROOTS:
        moved = archive_grid_root(args.repo_root / rel, args.dry_run)
        total.update(moved)
        print(f'{rel}: {sum(moved.values())} run dirs archived {dict(moved)}')
    whole = sum(archive_whole_dir(args.repo_root / rel, args.dry_run) for rel in WHOLE_DIRS)
    print(f'total: {sum(total.values())} run dirs {dict(total)}, {whole} whole trees')


if __name__ == '__main__':
    main()
