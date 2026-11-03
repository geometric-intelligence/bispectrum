#!/usr/bin/env python
"""Job manifest for the camera-ready PCam / OrganMNIST3D reruns across machines.

The jobs are the ``train.py`` calls of the existing pipelines, captured without training
(``run_camera_ready.sh capture``), so the sweep scripts stay the single definition of the
protocol. Each job gets a peak-memory estimate from ``train.py --memory_check`` (one
forward/backward step per distinct model configuration) and a duration estimate from the
wall times of earlier runs, then a machine:

    build     capture TSVs -> raw manifest (id = sha1 of cwd + argv)
    memory    run --memory_check once per configuration on the local GPUs -> JSONL
    costs     wall time per epoch of existing results.json files -> JSONL
    assign    memory + costs -> manifest.json: longest job first to the machine that would
              finish it earliest among those whose GPUs fit it (``--remaining`` reassigns
              only jobs that have not started, given the runners' state directories)
    validate  every job has a done marker and a current results.json (after collect)

Usage (from the repository root)::

    bash rebuttal/run_camera_ready.sh capture /tmp/cr_capture pcam pcam_12500 pcam_full \\
        pcam_matched organ3d
    python rebuttal/camera_ready_manifest.py build --capture_dir /tmp/cr_capture --out raw.json
    python rebuttal/camera_ready_manifest.py memory --manifest raw.json --gpus 0-7 --out mem.jsonl
    python rebuttal/camera_ready_manifest.py costs --scan experiments rebuttal/results --out c.jsonl
    python rebuttal/camera_ready_manifest.py assign --manifest raw.json --memory mem.jsonl \\
        --costs c.jsonl
    python rebuttal/camera_ready_manifest.py validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess  # nosec B404 - runs train.py --memory_check with a manifest argv
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'rebuttal'))

from gpu_job_runner import Job, parse_gpus, simulate  # noqa: E402

JOBS_DIR: Final = REPO_ROOT / 'rebuttal' / 'results' / 'camera_ready_jobs'
DEFAULT_MANIFEST: Final = JOBS_DIR / 'manifest.json'
GIB: Final = 2**30

# Peak reserved memory (memory_check) above this never goes to a 40 GB A100.
SMALL_GPU_MAX_PEAK_GIB: Final = 36.0
# Job memory for packing: memory_check's peak reserved memory plus slack for allocator
# growth over a run, plus the CUDA context, which torch does not count.
MEM_SLACK: Final = 1.05
CUDA_CONTEXT_GIB: Final = 1.5
# A configuration that runs out of memory on a 40 GB GPU is assumed to need an 80 GB GPU
# to itself.
OOM_MEM_GIB: Final = 60.0
DEFAULT_EST_HOURS: Final = 1.0


@dataclass(frozen=True)
class Machine:
    gpus: int
    gpu_mem_gib: float  # packing budget per GPU (job mem_gib includes the CUDA context)
    max_workers_per_gpu: int
    speed: float  # wall time relative to the cost estimates (A100, two jobs per GPU)
    start_h: float  # hours until the machine starts on manifest jobs

    @property
    def slots(self) -> int:
        return self.gpus * self.max_workers_per_gpu


# Speeds are rough (H100 ~1.6x an A100 on these models); start_h on the H100 is the
# Spherical MNIST pipeline that runs there first.
MACHINES: Final[dict[str, Machine]] = {
    'l8a100': Machine(gpus=8, gpu_mem_gib=39.0, max_workers_per_gpu=2, speed=1.0, start_h=0.0),
    'bongo': Machine(gpus=2, gpu_mem_gib=77.0, max_workers_per_gpu=2, speed=1.0, start_h=0.0),
    'h100': Machine(gpus=1, gpu_mem_gib=77.0, max_workers_per_gpu=2, speed=0.65, start_h=5.0),
}


# ---------------------------------------------------------------------------
# argv helpers
# ---------------------------------------------------------------------------
def flag(argv: Sequence[str], name: str, n: int = 1) -> list[str] | None:
    """Values following ``name`` in argv (n of them), or None if absent."""
    if name not in argv:
        return None
    i = list(argv).index(name)
    return list(argv[i + 1 : i + 1 + n])


def width_tag(dataset: str, argv: Sequence[str]) -> str:
    if dataset == 'organ3d':
        channels = flag(argv, '--channels', 2)
        return f'ch{"_".join(channels)}' if channels else 'ch4_8'
    bandlimit = flag(argv, '--bandlimit')
    if bandlimit:
        return f'bl{bandlimit[0]}'
    growth = flag(argv, '--growth_rate')
    return f'gr{growth[0]}' if growth else 'gr?'


def config_key(dataset: str, argv: Sequence[str]) -> str:
    """Everything that sets the memory footprint of a run (not seed, size or epochs)."""
    model = (flag(argv, '--model') or ['?'])[0]
    mode = (flag(argv, '--train_mode') or ['C'])[0]
    batch = (flag(argv, '--batch_size') or ['?'])[0]
    extra = [a for a in argv if a.startswith('--bispectrum_')]
    return '_'.join([dataset, model, width_tag(dataset, argv), mode, f'bs{batch}', *extra])


def job_label(pipeline: str, dataset: str, argv: Sequence[str]) -> str:
    model = (flag(argv, '--model') or ['?'])[0]
    mode = (flag(argv, '--train_mode') or ['C'])[0]
    size = (flag(argv, '--train_size') or ['full'])[0]
    seed = (flag(argv, '--seed') or ['?'])[0]
    prefix = pipeline if pipeline.startswith(dataset) else f'{dataset}_{pipeline}'
    return f'{prefix}_{model}_{width_tag(dataset, argv)}_{mode}_n{size}_seed{seed}'


def epoch_samples(dataset: str, argv: Sequence[str]) -> int:
    """Training samples per epoch under the floor (train.py's min_epoch_samples)."""
    size = flag(argv, '--train_size')
    if dataset == 'organ3d':
        return 971
    if size is None:
        return 262144
    return max(int(size[0]), 10000)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def parse_capture(path: Path) -> list[dict[str, Any]]:
    """One job per ``<cwd>\\t<shell-quoted argv>`` line written by the CAPTURE_JOBS hook."""
    pipeline = path.stem
    jobs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cwd, quoted = line.split('\t', 1)
        argv = shlex.split(quoted)
        parts = Path(cwd).parts
        cwd_rel = str(Path(*parts[parts.index('experiments') :]))
        dataset = Path(cwd_rel).name
        digest = hashlib.sha1(f'{cwd_rel}\0{" ".join(argv)}'.encode(), usedforsecurity=False)
        jobs.append(
            {
                'id': digest.hexdigest()[:12],
                'label': job_label(pipeline, dataset, argv),
                'pipeline': pipeline,
                'dataset': dataset,
                'config': config_key(dataset, argv),
                'cwd': cwd_rel,
                'argv': argv,
            }
        )
    return jobs


def build(capture_dir: Path) -> dict[str, Any]:
    jobs = [job for path in sorted(capture_dir.glob('*.tsv')) for job in parse_capture(path)]
    for key in ('id', 'label'):
        dupes = [v for v, n in Counter(j[key] for j in jobs).items() if n > 1]
        if dupes:
            raise SystemExit(f'duplicate job {key}s: {dupes[:5]}')
    return {
        'created': datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(),
        'pipelines': dict(Counter(j['pipeline'] for j in jobs)),
        'jobs': jobs,
    }


def _git_sha() -> str:
    out = subprocess.run(  # nosec B603 B607 - fixed git query
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
def memory_argv(job: dict[str, Any], out_dir: Path) -> list[str]:
    argv = list(job['argv'])
    i = argv.index('--output_dir')
    argv[i + 1] = str(out_dir / job['config'])
    return [*argv, '--memory_check', '--memory_headroom', '0.99']


def run_memory_checks(
    manifest: dict[str, Any], gpus: Sequence[int], out: Path, python: str
) -> list[dict[str, Any]]:
    """--memory_check once per configuration, one process per GPU at a time."""
    done = {r['config'] for r in read_jsonl(out)} if out.exists() else set()
    todo: dict[str, dict[str, Any]] = {}
    for job in manifest['jobs']:
        if job['config'] not in done:
            todo.setdefault(job['config'], job)
    print(f'{len(todo)} configurations to check ({len(done)} already in {out})', flush=True)
    queue = sorted(todo.values(), key=lambda j: j['config'])
    free = list(gpus)
    running: list[tuple[subprocess.Popen[str], int, dict[str, Any], float]] = []
    tmp = Path(tempfile.gettempdir()) / 'camera_ready_memory_check'
    while queue or running:
        while queue and free:
            job, gpu = queue.pop(0), free.pop(0)
            env = dict(os.environ) | {
                'CUDA_VISIBLE_DEVICES': str(gpu),
                'OMP_NUM_THREADS': '8',
                'PYTHONPATH': str(REPO_ROOT / 'src'),
            }
            proc = subprocess.Popen(  # nosec B603 - argv from the manifest, no shell
                [python, *memory_argv(job, tmp)],
                cwd=REPO_ROOT / job['cwd'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((proc, gpu, job, time.time()))
        time.sleep(2)
        for item in list(running):
            proc, gpu, job, started = item
            if proc.poll() is None:
                continue
            running.remove(item)
            free.append(gpu)
            output = proc.stdout.read() if proc.stdout else ''
            record = _last_json(output) or {'status': 'crash', 'error': output[-2000:]}
            record.update(
                config=job['config'], exit_code=proc.returncode, check_s=time.time() - started
            )
            with out.open('a') as f:
                f.write(json.dumps(record) + '\n')
            peak = record.get('max_memory_reserved_bytes', 0) / GIB
            print(f'GPU {gpu} {record["status"]:5s} {peak:6.1f} GiB  {job["config"]}', flush=True)
    return read_jsonl(out)


def _last_json(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def memory_table(records: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    """Config -> peak reserved GiB on the checking GPU, None if it ran out of memory."""
    table: dict[str, float | None] = {}
    for record in records:
        ok = record.get('status') == 'ok'
        table[record['config']] = record['max_memory_reserved_bytes'] / GIB if ok else None
    return table


def job_mem_gib(peak_gib: float | None) -> float:
    if peak_gib is None:
        return OOM_MEM_GIB
    return round(peak_gib * MEM_SLACK + CUDA_CONTEXT_GIB, 2)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------
def scan_costs(roots: Sequence[Path]) -> list[dict[str, Any]]:
    """Seconds per epoch of every results.json with a wall time under ``roots``."""
    records = []
    for root in roots:
        for path in sorted(root.rglob('results.json')):
            try:
                run = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            dataset = run.get('dataset')
            if (
                dataset not in ('pcam', 'organ3d')
                or not run.get('wall_time_s')
                or not run.get('epochs_run')
            ):
                continue
            argv = [str(a) for a in run.get('argv') or []]
            if not argv:
                continue
            records.append(
                {
                    'path': str(path),
                    'dataset': dataset,
                    'model': run.get('model'),
                    'width': width_tag(dataset, argv),
                    'batch_size': run.get('batch_size')
                    or (flag(argv, '--batch_size') or [None])[0],
                    # Runs from before the floor trained on the whole subset each epoch.
                    'epoch_samples': run.get('epoch_samples') or run.get('train_examples'),
                    'gpu_name': run.get('gpu_name') or '',
                    'sec_per_epoch': run['wall_time_s'] / run['epochs_run'],
                }
            )
    return records


def a100_sec_per_epoch(record: dict[str, Any]) -> float:
    """Normalize to an A100 (the H100 runs ~1.6x faster on these models)."""
    return record['sec_per_epoch'] * (1.6 if 'H100' in record['gpu_name'] else 1.0)


class CostModel:
    """Hours per job from the median seconds per epoch of the closest earlier runs."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.by_key: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
        for r in records:
            if not r.get('epoch_samples'):
                continue
            sample = (int(r['epoch_samples']), a100_sec_per_epoch(r))
            self.by_key[(r['dataset'], r['model'], r['width'], str(r['batch_size']))].append(
                sample
            )
            self.by_key[(r['dataset'], r['model'], r['width'])].append(sample)
            self.by_key[(r['dataset'], r['model'])].append(sample)
            self.by_key[(r['dataset'],)].append(sample)
        self.fallbacks: Counter[int] = Counter()

    def sec_per_epoch(self, dataset: str, argv: Sequence[str]) -> float | None:
        model = (flag(argv, '--model') or ['?'])[0]
        batch = (flag(argv, '--batch_size') or ['?'])[0]
        samples = epoch_samples(dataset, argv)
        keys = [
            (dataset, model, width_tag(dataset, argv), batch),
            (dataset, model, width_tag(dataset, argv)),
            (dataset, model),
            (dataset,),
        ]
        for level, key in enumerate(keys):
            matches = self.by_key.get(key, [])
            same = [s for n, s in matches if n == samples]
            if same:
                self.fallbacks[level] += 1
                return statistics.median(same)
            if matches:
                self.fallbacks[level + 10] += 1
                # Different samples per epoch: scale linearly (validation cost included).
                return statistics.median(s * samples / n for n, s in matches)
        self.fallbacks[99] += 1
        return None

    def est_hours(self, dataset: str, argv: Sequence[str]) -> float:
        epochs = int((flag(argv, '--epochs') or ['1'])[0])
        sec = self.sec_per_epoch(dataset, argv)
        return DEFAULT_EST_HOURS if sec is None else round(sec * epochs / 3600, 3)


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------
def eligible(job: Job, peak_gib: float | None, machine: str) -> bool:
    spec = MACHINES[machine]
    if spec.gpu_mem_gib < 40 and (peak_gib is None or peak_gib > SMALL_GPU_MAX_PEAK_GIB):
        return False
    return job.mem_gib <= spec.gpu_mem_gib


def assign(
    jobs: Sequence[Job],
    peaks: dict[str, float | None],
    configs: dict[str, str],
    machines: dict[str, Machine],
    fixed: dict[str, str] | None = None,
) -> dict[str, str]:
    """Job id -> machine: longest first to the machine with the earliest projected finish.

    ``fixed`` (job id -> machine) are jobs already done or running; they count toward their
    machine's load only through ``start_h``, which the caller sets to the remaining time.
    """
    fixed = fixed or {}
    load = dict.fromkeys(machines, 0.0)
    placement: dict[str, str] = dict(fixed)
    for job in sorted(jobs, key=lambda j: (-j.est_hours, j.label)):
        if job.id in fixed:
            continue
        options = [m for m in machines if eligible(job, peaks.get(configs[job.id]), m)]
        if not options:
            raise SystemExit(f'no machine fits {job.label} ({job.mem_gib} GiB)')

        def finish(m: str, job: Job = job) -> float:
            spec = machines[m]
            hours = job.est_hours * spec.speed
            return spec.start_h + max((load[m] + hours) / spec.slots, hours)

        best = min(options, key=lambda m: (finish(m), m))
        load[best] += job.est_hours * machines[best].speed
        placement[job.id] = best
    return placement


def report(jobs: Sequence[Job], machines: dict[str, Machine]) -> None:
    print(
        f'{"machine":8s} {"jobs":>5s} {"est h":>8s} {"slots":>5s} {"start":>6s} {"simulated finish":>17s}'
    )
    for name, spec in machines.items():
        mine = [j for j in jobs if j.machine == name]
        hours = sum(j.est_hours for j in mine) * spec.speed
        makespan = simulate(
            mine, spec.gpus, spec.gpu_mem_gib, spec.max_workers_per_gpu, spec.speed
        )
        print(
            f'{name:8s} {len(mine):5d} {hours:8.1f} {spec.slots:5d} {spec.start_h:6.1f} '
            f'{spec.start_h + makespan:15.1f} h'
        )
    longest = sorted(jobs, key=lambda j: -j.est_hours)[:5]
    print(
        'longest jobs: '
        + ', '.join(f'{j.label} {j.est_hours:.1f} h ({j.machine})' for j in longest)
    )
    by_pipeline: dict[str, Counter[str]] = defaultdict(Counter)
    for j in jobs:
        by_pipeline[j.pipeline][j.machine] += 1
    for pipeline, counts in sorted(by_pipeline.items()):
        print(f'  {pipeline:14s} ' + ' '.join(f'{m}={n}' for m, n in sorted(counts.items())))


def started_jobs(state_root: Path) -> tuple[set[str], dict[str, str]]:
    """(done job ids, running job label -> machine) from the runners' state directories."""
    done = {p.stem for p in state_root.glob('*/done/*.json')}
    running: dict[str, str] = {}
    for progress in state_root.glob('*/progress.json'):
        state = json.loads(progress.read_text())
        for gpu in state.get('gpus', {}).values():
            for label in gpu.get('jobs', []):
                running[label] = progress.parent.name
    return done, running


def build_manifest(
    raw: dict[str, Any],
    peaks: dict[str, float | None],
    costs: CostModel,
    machines: dict[str, Machine],
    previous: dict[str, Any] | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    jobs = [
        Job(
            id=j['id'],
            label=j['label'],
            cwd=j['cwd'],
            argv=tuple(j['argv']),
            mem_gib=job_mem_gib(peaks.get(j['config'])),
            est_hours=costs.est_hours(j['dataset'], j['argv']),
            pipeline=j['pipeline'],
        )
        for j in raw['jobs']
    ]
    configs = {j['id']: j['config'] for j in raw['jobs']}
    missing = sorted({c for c in configs.values() if c not in peaks})
    if missing:
        raise SystemExit(
            f'no memory_check record for {len(missing)} configurations: {missing[:5]}'
        )
    fixed: dict[str, str] = {}
    if previous is not None and state_root is not None:
        before = {j['id']: j['machine'] for j in previous['jobs']}
        done, running = started_jobs(state_root)
        fixed = {j.id: before[j.id] for j in jobs if j.id in done and j.id in before}
        fixed |= {j.id: running[j.label] for j in jobs if j.label in running}
    placement = assign(jobs, peaks, configs, machines, fixed)
    jobs = [Job(**{**asdict(j), 'machine': placement[j.id]}) for j in jobs]
    report([j for j in jobs if j.id not in fixed], machines)
    return {
        'created': datetime.now().isoformat(timespec='seconds'),
        'git_sha': raw.get('git_sha', ''),
        'machines': {
            name: {**asdict(spec), 'slots': spec.slots} for name, spec in machines.items()
        },
        'pipelines': raw.get('pipelines', {}),
        'jobs': [
            {
                **j.to_dict(),
                'dataset': r['dataset'],
                'config': r['config'],
                'peak_gib': peaks[r['config']],
            }
            for j, r in zip(jobs, raw['jobs'], strict=True)
        ],
    }


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def validate(manifest: dict[str, Any], state_root: Path) -> int:
    markers = {p.stem: p for p in state_root.glob('*/done/*.json')}
    problems: list[str] = []
    cells: Counter[str] = Counter()
    for job in manifest['jobs']:
        marker = markers.get(job['id'])
        if marker is None:
            problems.append(f'not done: {job["label"]} ({job["machine"]})')
            continue
        done = json.loads(marker.read_text())
        results = REPO_ROOT / (done.get('results_json') or '')
        if not done.get('results_json') or not results.is_file():
            problems.append(f'no results.json: {job["label"]} ({done.get("host")})')
            continue
        run = json.loads(results.read_text())
        issues = []
        if run.get('early_stopping_patience') != 0:
            issues.append(f'patience={run.get("early_stopping_patience")}')
        if run.get('epochs_run') != run.get('epochs'):
            issues.append(f'epochs_run={run.get("epochs_run")} != {run.get("epochs")}')
        if (
            run.get('model') == 'bispectrum'
            and (run.get('bispectrum_head') or {}).get('nonlin') != 'post'
        ):
            issues.append(f'bispectrum_head={run.get("bispectrum_head")}')
        if not run.get('git_sha'):
            issues.append('no git_sha')
        if issues:
            problems.append(f'{job["label"]}: ' + ', '.join(issues))
        cells[job['label'].rsplit('_seed', 1)[0]] += 1
    seeds = Counter(cells.values())
    print(f'{len(manifest["jobs"])} jobs, {len(markers)} done markers, {len(cells)} cells')
    print('cells by seed count: ' + ', '.join(f'{n} seeds: {k}' for n, k in sorted(seeds.items())))
    for problem in problems:
        print('  ' + problem)
    print(f'{len(problems)} problems')
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build')
    p.add_argument('--capture_dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=JOBS_DIR / 'raw.json')

    p = sub.add_parser('memory')
    p.add_argument('--manifest', type=Path, default=JOBS_DIR / 'raw.json')
    p.add_argument('--gpus', default='0')
    p.add_argument('--out', type=Path, default=JOBS_DIR / 'memory.jsonl')
    p.add_argument('--python', default=sys.executable)

    p = sub.add_parser('costs')
    p.add_argument('--scan', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, default=JOBS_DIR / 'costs.jsonl')

    p = sub.add_parser('assign')
    p.add_argument('--manifest', type=Path, default=JOBS_DIR / 'raw.json')
    p.add_argument('--memory', type=Path, default=JOBS_DIR / 'memory.jsonl')
    p.add_argument('--costs', type=Path, nargs='+', default=[JOBS_DIR / 'costs.jsonl'])
    p.add_argument('--out', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        '--remaining',
        action='store_true',
        help='keep done/running jobs where they are; reassign the rest (reads --out and --state_root)',
    )
    p.add_argument('--state_root', type=Path, default=JOBS_DIR / 'state')
    p.add_argument(
        '--start_h',
        nargs='*',
        default=[],
        metavar='MACHINE=HOURS',
        help='hours until a machine is free for new jobs (e.g. h100=0 once its SMNIST is done)',
    )
    p.add_argument('--exclude', nargs='*', default=[], help='machines to leave out')

    p = sub.add_parser('validate')
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--state_root', type=Path, default=JOBS_DIR / 'state')

    args = parser.parse_args(argv)

    if args.command == 'build':
        raw = build(args.capture_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(raw, indent=1) + '\n')
        print(
            f'{len(raw["jobs"])} jobs {raw["pipelines"]}, {len({j["config"] for j in raw["jobs"]})} configs'
        )
        return 0
    if args.command == 'memory':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        records = run_memory_checks(
            json.loads(args.manifest.read_text()), parse_gpus(args.gpus), args.out, args.python
        )
        failed = [r['config'] for r in records if r.get('status') != 'ok']
        print(f'{len(records)} records, not ok on this GPU: {failed}')
        return 0
    if args.command == 'costs':
        records = scan_costs(args.scan)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(''.join(json.dumps(r) + '\n' for r in records))
        print(f'{len(records)} cost records -> {args.out}')
        return 0
    if args.command == 'assign':
        raw = json.loads(args.manifest.read_text())
        peaks = memory_table(read_jsonl(args.memory))
        costs = CostModel([r for path in args.costs for r in read_jsonl(path)])
        machines = {m: s for m, s in MACHINES.items() if m not in args.exclude}
        for item in args.start_h:
            name, hours = item.split('=')
            machines[name] = Machine(**{**asdict(machines[name]), 'start_h': float(hours)})
        previous = json.loads(args.out.read_text()) if args.remaining else None
        manifest = build_manifest(
            raw, peaks, costs, machines, previous, args.state_root if args.remaining else None
        )
        print(
            f'cost fallbacks (0 = exact config, 10+ = scaled samples/epoch, 99 = default): {dict(costs.fallbacks)}'
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=1) + '\n')
        tmp.replace(args.out)
        print(f'-> {args.out}')
        return 0
    if args.command == 'validate':
        return validate(json.loads(args.manifest.read_text()), args.state_root)
    return 1


if __name__ == '__main__':
    sys.exit(main())
