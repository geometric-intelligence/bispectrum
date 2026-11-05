#!/usr/bin/env python
"""Bispectral pooling head ablation: 90 training runs scheduled over the local GPUs.

The PCam and OrganMNIST3D bispectrum models used to apply a ReLU to the signed
bispectrum *before* the 1x1 projection (``--bispectrum_head pre``), which discards the
sign/phase information of the bispectrum. This ablation compares that head with the
fixed one and two opt-in feature variants, against the strongest baselines, under the
camera-ready protocol (fixed epoch budget at an SGDR cycle end, no early stopping, one
batch size per dataset):

    pre_log          --bispectrum_head pre                        (original head)
    post_log         (default)                                    (ReLU after the projection)
    post_log_power   --bispectrum_power                           (+ power spectrum)
    post_cbrt_power  --bispectrum_power --bispectrum_scale cbrt   (+ power, cube-root scale)

OrganMNIST3D (70 epochs, batch 16, mode C, 3 seeds): channels (4, 8) at n = 100 and the
full set, channels (16, 32) on the full set; the four heads plus max_pool and norm_pool.
PCam (30 epochs, batch 64, 3 seeds, rotation evaluation on): n = 100 and n = 12,500; the
four heads (gr 4, mode C), norm_pool (gr 4, mode C) and standard (gr 12, mode R).

Jobs run as independent ``train.py`` processes, ``--workers_per_gpu`` at a time per GPU,
longest first. A job whose output directory already holds a ``results.json`` is skipped,
so the runner can be restarted. Each job logs to ``<results>/logs/<job>.log``; the runner
writes a progress line to ``<results>/logs/runner.log`` every ten minutes. SIGINT/SIGTERM
terminate every running job (whole process groups). Exit status is non-zero if any job
failed.

Usage (from the repository root, inside tmux via ``experiments/tooling/run_head_ablation.sh``)::

    .venv/bin/python experiments/tooling/run_head_ablation.py --gpus 0-7 --workers_per_gpu 2
    .venv/bin/python experiments/tooling/run_head_ablation.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess  # nosec B404 - launches train.py with a fixed argv
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, Literal

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS: Final = REPO_ROOT / 'experiments' / 'tooling' / 'results' / 'head_ablation'

Dataset = Literal['organ3d', 'pcam']

SEEDS: Final = (42, 123, 456)

# label -> extra train.py flags for the bispectrum model
HEAD_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    'pre_log': ('--bispectrum_head', 'pre'),
    'post_log': (),
    'post_log_power': ('--bispectrum_power',),
    'post_cbrt_power': ('--bispectrum_power', '--bispectrum_scale', 'cbrt'),
}

ORGAN3D_COMMON: Final = (
    '--train_mode', 'C', '--epochs', '70', '--batch_size', '16', '--skip_rotation',
    '--data_dir', './organ3d_data',
)  # fmt: skip
ORGAN3D_BASELINES: Final = ('max_pool', 'norm_pool')
# setting label -> (channels, train_size, relative cost used for scheduling)
ORGAN3D_SETTINGS: Final[dict[str, tuple[tuple[int, int], int | None, float]]] = {
    'ch4_8_n100': ((4, 8), 100, 1.0),
    'ch4_8_full': ((4, 8), None, 1.0),
    'ch16_32_full': ((16, 32), None, 4.0),
}

PCAM_COMMON: Final = ('--epochs', '30', '--batch_size', '64', '--data_dir', './pcam_data')
# label -> (model, growth rate, train mode)
PCAM_BASELINES: Final[dict[str, tuple[str, int, str]]] = {
    'norm_pool': ('norm_pool', 4, 'C'),
    'standard': ('standard', 12, 'R'),
}
PCAM_SETTINGS: Final[dict[str, tuple[int, float]]] = {'n100': (100, 3.0), 'n12500': (12500, 4.0)}


@dataclass(frozen=True)
class Job:
    dataset: Dataset
    setting: str
    label: str
    seed: int
    argv: tuple[str, ...]
    cost: float

    @property
    def name(self) -> str:
        return f'{self.dataset}_{self.setting}_{self.label}_seed{self.seed}'

    def out_dir(self, results_root: Path) -> Path:
        return results_root / self.dataset / self.setting / self.label / f'seed{self.seed}'

    def results_json(self, results_root: Path) -> Path | None:
        found = sorted(self.out_dir(results_root).glob('*/results.json'))
        return found[0] if found else None

    def command(self, python: str, results_root: Path) -> list[str]:
        return [python, 'train.py', *self.argv, '--output_dir', str(self.out_dir(results_root))]

    @property
    def cwd(self) -> Path:
        return REPO_ROOT / 'experiments' / self.dataset


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for setting, (channels, train_size, cost) in ORGAN3D_SETTINGS.items():
        size = ('--train_size', str(train_size)) if train_size else ()
        base = ('--channels', *map(str, channels), *ORGAN3D_COMMON, *size)
        for label, flags in HEAD_VARIANTS.items():
            for seed in SEEDS:
                argv = ('--model', 'bispectrum', *flags, *base, '--seed', str(seed))
                jobs.append(Job('organ3d', setting, label, seed, argv, cost))
        for model in ORGAN3D_BASELINES:
            for seed in SEEDS:
                argv = ('--model', model, *base, '--seed', str(seed))
                jobs.append(Job('organ3d', setting, model, seed, argv, cost))
    for setting, (train_size, cost) in PCAM_SETTINGS.items():
        base = (*PCAM_COMMON, '--train_size', str(train_size))
        for label, flags in HEAD_VARIANTS.items():
            for seed in SEEDS:
                argv = (
                    '--model', 'bispectrum', '--growth_rate', '4', '--train_mode', 'C',
                    *flags, *base, '--seed', str(seed),
                )  # fmt: skip
                jobs.append(Job('pcam', setting, label, seed, argv, cost))
        for label, (model, gr, mode) in PCAM_BASELINES.items():
            for seed in SEEDS:
                argv = (
                    '--model', model, '--growth_rate', str(gr), '--train_mode', mode,
                    *base, '--seed', str(seed),
                )  # fmt: skip
                jobs.append(Job('pcam', setting, label, seed, argv, cost))
    assert len(jobs) == 90, len(jobs)
    assert len({j.name for j in jobs}) == len(jobs)
    return jobs


def parse_gpus(spec: str) -> list[int]:
    gpus: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = (int(x) for x in part.split('-'))
            gpus.extend(range(lo, hi + 1))
        elif part:
            gpus.append(int(part))
    if not gpus:
        raise ValueError(f'no GPUs in {spec!r}')
    return gpus


@dataclass
class Running:
    job: Job
    gpu: int
    proc: subprocess.Popen[bytes]
    started: float
    log: Path


@dataclass
class Runner:
    jobs: list[Job]
    gpus: list[int]
    workers_per_gpu: int
    results_root: Path
    python: str
    omp_threads: int
    progress_every_s: float = 600.0
    running: list[Running] = field(default_factory=list)
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, int] = field(default_factory=dict)
    t0: float = 0.0
    _stop: bool = False

    @property
    def log_dir(self) -> Path:
        return self.results_root / 'logs'

    def log(self, msg: str) -> None:
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.log_dir / 'runner.log', 'a') as f:
            f.write(line + '\n')

    def _free_slots(self) -> Iterator[int]:
        load = dict.fromkeys(self.gpus, 0)
        for r in self.running:
            load[r.gpu] += 1
        for gpu in self.gpus:
            for _ in range(self.workers_per_gpu - load[gpu]):
                yield gpu

    def _start(self, job: Job, gpu: int) -> None:
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS=str(self.omp_threads),
            PYTHONUNBUFFERED='1',
            PYTHONPATH=f'{REPO_ROOT / "src"}:{env.get("PYTHONPATH", "")}'.rstrip(':'),
        )
        job.out_dir(self.results_root).mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f'{job.name}.log'
        cmd = job.command(self.python, self.results_root)
        with open(log, 'a') as f:
            f.write(
                f'# {time.strftime("%Y-%m-%d %H:%M:%S")} gpu={gpu} cwd={job.cwd}\n# {" ".join(cmd)}\n'
            )
            f.flush()
            proc = subprocess.Popen(  # nosec B603 - fixed argv built above, no shell
                cmd,
                cwd=job.cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.running.append(Running(job, gpu, proc, time.time(), log))
        self.log(f'START {job.name} on gpu {gpu} (pid {proc.pid})')

    def _reap(self) -> None:
        still: list[Running] = []
        for r in self.running:
            code = r.proc.poll()
            if code is None:
                still.append(r)
                continue
            minutes = (time.time() - r.started) / 60
            if code == 0 and r.job.results_json(self.results_root) is not None:
                self.done.append(r.job.name)
                self.log(f'DONE  {r.job.name} in {minutes:.1f} min')
            else:
                self.failed[r.job.name] = code
                self.log(f'FAIL  {r.job.name} exit {code} after {minutes:.1f} min (see {r.log})')
        self.running = still

    def _terminate_all(self, signum: int, _frame: object) -> None:
        self._stop = True
        self.log(f'signal {signum}: terminating {len(self.running)} running jobs')
        for r in self.running:
            try:
                os.killpg(r.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _progress(self) -> None:
        total = len(self.jobs)
        finished = len(self.done) + len(self.skipped) + len(self.failed)
        elapsed = (time.time() - self.t0) / 3600
        active = ', '.join(f'{r.job.name}@{r.gpu}' for r in self.running)
        self.log(
            f'PROGRESS {finished}/{total} finished ({len(self.done)} run, {len(self.skipped)} skipped, '
            f'{len(self.failed)} failed), {len(self.running)} running, {elapsed:.2f} h elapsed; {active}'
        )

    def run(self) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        signal.signal(signal.SIGINT, self._terminate_all)
        signal.signal(signal.SIGTERM, self._terminate_all)

        queue: list[Job] = []
        for job in sorted(self.jobs, key=lambda j: -j.cost):
            if job.results_json(self.results_root) is not None:
                self.skipped.append(job.name)
            else:
                queue.append(job)
        self.log(
            f'{len(self.jobs)} jobs: {len(queue)} to run, {len(self.skipped)} already done; '
            f'gpus {self.gpus} x {self.workers_per_gpu} workers, OMP_NUM_THREADS={self.omp_threads}'
        )
        last_progress = time.time()
        while (queue or self.running) and not self._stop:
            self._reap()
            for gpu in self._free_slots():
                if not queue:
                    break
                self._start(queue.pop(0), gpu)
            if time.time() - last_progress >= self.progress_every_s:
                self._progress()
                last_progress = time.time()
            time.sleep(5)
        while self.running:
            self._reap()
            time.sleep(1)
        self._progress()
        if self.failed:
            self.log('FAILED: ' + ', '.join(f'{n} (exit {c})' for n, c in self.failed.items()))
        status = 'INTERRUPTED' if self._stop else 'COMPLETE'
        self.log(
            f'{status}: {len(self.done)} run, {len(self.skipped)} skipped, {len(self.failed)} failed'
        )
        return 1 if self.failed or self._stop else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--gpus', default='0-7', help='GPU indices, e.g. "0-7" or "0,2,5"')
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--omp_threads', type=int, default=8)
    parser.add_argument('--results_root', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--only', default='', help='substring filter on job names (smoke tests)')
    parser.add_argument('--dry_run', action='store_true', help='print the job manifest and exit')
    args = parser.parse_args()

    jobs = [j for j in build_jobs() if args.only in j.name]
    results_root: Path = args.results_root.resolve()
    if args.dry_run:
        manifest = [
            {**asdict(j), 'name': j.name, 'out_dir': str(j.out_dir(results_root)),
             'done': j.results_json(results_root) is not None}
            for j in sorted(jobs, key=lambda j: -j.cost)
        ]  # fmt: skip
        print(json.dumps(manifest, indent=1))
        print(
            f'{len(jobs)} jobs, {sum(m["done"] for m in manifest)} already done', file=sys.stderr
        )
        return 0
    runner = Runner(
        jobs=jobs,
        gpus=parse_gpus(args.gpus),
        workers_per_gpu=args.workers_per_gpu,
        results_root=results_root,
        python=args.python,
        omp_threads=args.omp_threads,
    )
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
