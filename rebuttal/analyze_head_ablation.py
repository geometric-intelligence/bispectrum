#!/usr/bin/env python
"""Summarize the bispectral pooling head ablation (``rebuttal/run_head_ablation.py``).

Reads every ``results.json`` under ``<results>/<dataset>/<setting>/<label>/seed*/`` and
writes ``summary.md`` and ``summary.json`` next to them:

* per dataset / setting / label: mean +- std (ddof=1) over seeds of the primary test
  metric (accuracy for OrganMNIST3D, AUC for PCam) at the best-validation checkpoint, the
  same metric at the last epoch (``test_final``), the best validation AUC, parameters and
  wall time;
* PCam only: the rotation-robustness AUC split into exact rotations (multiples of 90
  degrees, even group elements) and 45-degree rotations (odd elements, bilinear
  interpolation), which separates equivariance error from interpolation error;
* Welch t-test p-values of every head against the original head (``pre_log``) and
  against the best baseline of the setting;
* the head decision: the bispectrum variant with the highest mean best-validation AUC,
  per setting and overall (mean rank across settings), without looking at test metrics.

Usage::

    .venv/bin/python rebuttal/analyze_head_ablation.py [--results_root DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scipy import stats

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS: Final = REPO_ROOT / 'rebuttal' / 'results' / 'head_ablation'

HEAD_LABELS: Final = ('pre_log', 'post_log', 'post_log_power', 'post_cbrt_power')
PRIMARY_METRIC: Final = {'organ3d': 'accuracy', 'pcam': 'auc'}
LABEL_ORDER: Final = (*HEAD_LABELS, 'max_pool', 'norm_pool', 'standard')


@dataclass(frozen=True)
class Run:
    dataset: str
    setting: str
    label: str
    seed: int
    primary: float  # test metric at the best-validation checkpoint
    final: float  # test metric at the last epoch
    best_val_auc: float
    n_params: int
    wall_time_min: float
    epochs_run: int
    exact_auc: float | None  # PCam: mean AUC over rotations by multiples of 90 degrees
    odd_auc: float | None  # PCam: mean AUC over 45-degree rotations
    head: dict[str, Any] | None


def _rotation_split(rot: dict[str, Any]) -> tuple[float | None, float | None]:
    elements = {
        int(k[1:]): v['auc'] for k, v in rot.items() if k.startswith('g') and k[1:].isdigit()
    }
    if not elements:
        return None, None
    even = [v for k, v in elements.items() if k % 2 == 0]
    odd = [v for k, v in elements.items() if k % 2 == 1]
    return (statistics.fmean(even) if even else None, statistics.fmean(odd) if odd else None)


def load_runs(results_root: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(results_root.glob('*/*/*/seed*/*/results.json')):
        dataset, setting, label, seed_dir = path.relative_to(results_root).parts[:4]
        r = json.loads(path.read_text())
        metric = PRIMARY_METRIC[dataset]
        exact, odd = _rotation_split(r.get('rotation_robustness') or {})
        runs.append(
            Run(
                dataset=dataset,
                setting=setting,
                label=label,
                seed=int(seed_dir.removeprefix('seed')),
                primary=float(r['test_c'][metric]),
                final=float(r['test_final'][metric]),
                best_val_auc=float(r['best_val_auc']),
                n_params=int(r['n_params']),
                wall_time_min=float(r.get('wall_time_s', 0.0)) / 60,
                epochs_run=int(r.get('epochs_run', 0)),
                exact_auc=exact,
                odd_auc=odd,
                head=r.get('bispectrum_head'),
            )
        )
    return runs


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _welch_p(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    if statistics.pstdev(a) == 0 and statistics.pstdev(b) == 0:
        return 1.0 if statistics.fmean(a) == statistics.fmean(b) else 0.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def summarize(runs: list[Run]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, list[Run]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        groups[(run.dataset, run.setting)][run.label].append(run)

    settings: list[dict[str, Any]] = []
    for (dataset, setting), by_label in sorted(groups.items()):
        rows: dict[str, dict[str, Any]] = {}
        for label, rs in by_label.items():
            primary = [r.primary for r in rs]
            mean, std = _mean_std(primary)
            fmean, fstd = _mean_std([r.final for r in rs])
            vmean, vstd = _mean_std([r.best_val_auc for r in rs])
            row: dict[str, Any] = {
                'n': len(rs),
                'seeds': sorted(r.seed for r in rs),
                'primary_mean': mean,
                'primary_std': std,
                'final_mean': fmean,
                'final_std': fstd,
                'best_val_auc_mean': vmean,
                'best_val_auc_std': vstd,
                'n_params': rs[0].n_params,
                'wall_time_min_mean': statistics.fmean(r.wall_time_min for r in rs),
                'epochs_run': sorted({r.epochs_run for r in rs}),
                'head': rs[0].head,
                'primary_values': primary,
            }
            if any(r.exact_auc is not None for r in rs):
                row['exact_auc_mean'], row['exact_auc_std'] = _mean_std(
                    [r.exact_auc for r in rs if r.exact_auc is not None]
                )
                row['odd_auc_mean'], row['odd_auc_std'] = _mean_std(
                    [r.odd_auc for r in rs if r.odd_auc is not None]
                )
            rows[label] = row
        baselines = [lab for lab in rows if lab not in HEAD_LABELS]
        best_baseline = (
            max(baselines, key=lambda lab: rows[lab]['primary_mean']) if baselines else None
        )
        for label, row in rows.items():
            row['p_vs_pre_log'] = (
                _welch_p(row['primary_values'], rows['pre_log']['primary_values'])
                if 'pre_log' in rows and label != 'pre_log'
                else None
            )
            row['p_vs_best_baseline'] = (
                _welch_p(row['primary_values'], rows[best_baseline]['primary_values'])
                if best_baseline and label != best_baseline
                else None
            )
        heads = [lab for lab in HEAD_LABELS if lab in rows]
        val_winner = max(heads, key=lambda lab: rows[lab]['best_val_auc_mean']) if heads else None
        settings.append(
            {
                'dataset': dataset,
                'setting': setting,
                'metric': PRIMARY_METRIC[dataset],
                'best_baseline': best_baseline,
                'validation_winner': val_winner,
                'rows': rows,
            }
        )

    # Head decision: mean rank of best-validation AUC across settings (1 = best).
    ranks: dict[str, list[int]] = defaultdict(list)
    for s in settings:
        heads = sorted(
            (lab for lab in HEAD_LABELS if lab in s['rows']),
            key=lambda lab: -s['rows'][lab]['best_val_auc_mean'],
        )
        for rank, lab in enumerate(heads, start=1):
            ranks[lab].append(rank)
    mean_rank = {lab: statistics.fmean(r) for lab, r in ranks.items()}
    decision = (
        min(mean_rank, key=lambda lab: (mean_rank[lab], HEAD_LABELS.index(lab)))
        if mean_rank
        else None
    )
    return {
        'n_runs': len(runs),
        'settings': settings,
        'head_mean_validation_rank': mean_rank,
        'head_decision': decision,
    }


def _fmt(mean: float, std: float, pct: bool) -> str:
    if math.isnan(mean):
        return '-'
    scale = 100 if pct else 1
    return f'{mean * scale:.{1 if pct else 4}f} ± {std * scale:.{1 if pct else 4}f}'


def _fmt_p(p: float | None) -> str:
    return '-' if p is None else (f'{p:.3f}' if p >= 0.001 else '<0.001')


def render_markdown(summary: dict[str, Any]) -> str:
    out: list[str] = ['# Bispectral pooling head ablation', '']
    out.append(
        f'{summary["n_runs"]} runs. Mean ± std over seeds (ddof=1). Primary metric: test accuracy '
        '(OrganMNIST3D) or test AUC (PCam) at the best-validation checkpoint; `final` is the same '
        'metric at the last epoch. `p(pre)` / `p(base)`: Welch t-test against the original head '
        '(`pre_log`) and against the best baseline of the setting.'
    )
    out.append('')
    for s in summary['settings']:
        pct = s['metric'] == 'accuracy'
        out.append(f'## {s["dataset"]} / {s["setting"]}  (metric: {s["metric"]})')
        out.append('')
        has_rot = any('exact_auc_mean' in row for row in s['rows'].values())
        header = [
            'model',
            'n',
            'test (best val)',
            'test (final)',
            'best val AUC',
            'params',
            'min/run',
            'p(pre)',
            'p(base)',
        ]
        if has_rot:
            header[3:3] = ['AUC exact rot', 'AUC 45° rot']
        out.append('| ' + ' | '.join(header) + ' |')
        out.append('|' + '---|' * len(header))
        for label in sorted(
            s['rows'], key=lambda lab: LABEL_ORDER.index(lab) if lab in LABEL_ORDER else 99
        ):
            row = s['rows'][label]
            cells = [
                f'`{label}`' + (' ★' if label == s['validation_winner'] else ''),
                str(row['n']),
                _fmt(row['primary_mean'], row['primary_std'], pct),
                _fmt(row['final_mean'], row['final_std'], pct),
                f'{row["best_val_auc_mean"]:.4f} ± {row["best_val_auc_std"]:.4f}',
                f'{row["n_params"]:,}',
                f'{row["wall_time_min_mean"]:.0f}',
                _fmt_p(row['p_vs_pre_log']),
                _fmt_p(row['p_vs_best_baseline']),
            ]
            if has_rot:
                cells[3:3] = [
                    _fmt(
                        row.get('exact_auc_mean', math.nan),
                        row.get('exact_auc_std', math.nan),
                        False,
                    ),
                    _fmt(
                        row.get('odd_auc_mean', math.nan), row.get('odd_auc_std', math.nan), False
                    ),
                ]
            out.append('| ' + ' | '.join(cells) + ' |')
        out.append('')
        out.append(
            f'Best baseline: `{s["best_baseline"]}`. Validation winner among heads: `{s["validation_winner"]}` (★).'
        )
        out.append('')
    out.append('## Head decision (best-validation AUC, no test metrics involved)')
    out.append('')
    for lab, rank in sorted(summary['head_mean_validation_rank'].items(), key=lambda kv: kv[1]):
        out.append(f'- `{lab}`: mean rank {rank:.2f}')
    out.append('')
    out.append(f'**Decision: `{summary["head_decision"]}`**')
    out.append('')
    return '\n'.join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results_root: Path = args.results_root.resolve()

    runs = load_runs(results_root)
    if not runs:
        print(f'no results under {results_root}', file=sys.stderr)
        return 1
    summary = summarize(runs)
    (results_root / 'summary.json').write_text(json.dumps(summary, indent=1))
    markdown = render_markdown(summary)
    (results_root / 'summary.md').write_text(markdown)
    print(markdown)
    return 0


if __name__ == '__main__':
    sys.exit(main())
