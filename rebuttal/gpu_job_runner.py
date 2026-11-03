#!/usr/bin/env python
"""Run one machine's share of a job manifest on all of its GPUs.

The manifest (``rebuttal/camera_ready_manifest.py``) lists ``train.py`` invocations with a
peak-memory estimate (``mem_gib``), a duration estimate (``est_hours``) and the machine they
are assigned to; ``machines[<name>]`` gives the per-GPU memory budget and the number of
concurrent jobs per GPU. Jobs start longest first on the GPU with the fewest running jobs
whose running memory plus the job's stays within the budget, so two small runs share a
40 GB A100 while a 48 GiB run waits for an 80 GB GPU of its own. When the longest queued job
fits nowhere, the GPU expected to drain first is reserved for it: other jobs start there only
if their estimate ends before that GPU drains (EASY backfilling).

State lives in ``--state_dir``:

    done/<id>.json      exit code, host, GPU, git SHA, wall time, results.json path
    failed/<id>.json    last failure (removed once the job succeeds)
    logs/<label>.log    train.py output, one header line per attempt
    progress.json       counts and per-GPU jobs, rewritten every ``--progress_every_s``

A job with a done marker is skipped, so the runner can be restarted. A failed job is retried
``--retries`` times alone on a GPU (OOM from a bad memory estimate). The manifest is re-read
when it changes (``camera_ready_manifest.py assign --remaining``): newly assigned jobs are
queued, jobs moved to another machine leave the queue, running jobs are not interrupted.
SIGINT/SIGTERM/SIGHUP kill every running job's process group. Exit status is non-zero if any job
failed.

Usage (inside tmux via ``rebuttal/run_camera_ready.sh launch_jobs <machine>``)::

    .venv/bin/python rebuttal/gpu_job_runner.py --manifest M --machine l8a100 --state_dir S
    .venv/bin/python rebuttal/gpu_job_runner.py --manifest M --machine l8a100 --dry_run
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import signal
import socket
import subprocess  # nosec B404 - launches train.py with an argv from the manifest
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    id: str
    label: str
    cwd: str  # relative to the repository root
    argv: tuple[str, ...]  # train.py and its arguments
    mem_gib: float
    est_hours: float
    machine: str = ''
    pipeline: str = ''

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Job:
        return cls(
            id=record['id'],
            label=record['label'],
            cwd=record['cwd'],
            argv=tuple(record['argv']),
            mem_gib=float(record['mem_gib']),
            est_hours=float(record['est_hours']),
            machine=record.get('machine', ''),
            pipeline=record.get('pipeline', ''),
        )

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record['argv'] = list(self.argv)
        return record


@dataclass
class GpuSlots:
    index: int
    jobs: list[str] = field(default_factory=list)
    mem_gib: float = 0.0
    ends: dict[str, float] = field(default_factory=dict)  # job id -> estimated end (hours)

    def add(self, job: Job, end: float) -> None:
        self.jobs.append(job.id)
        self.mem_gib += job.mem_gib
        self.ends[job.id] = end

    def remove(self, job: Job) -> None:
        self.jobs.remove(job.id)
        self.mem_gib -= job.mem_gib
        self.ends.pop(job.id, None)


def fits(job: Job, gpu: GpuSlots, gpu_mem_gib: float, max_workers: int) -> bool:
    # The tolerance absorbs float residue left in mem_gib after jobs finish.
    return len(gpu.jobs) < max_workers and gpu.mem_gib + job.mem_gib <= gpu_mem_gib + 1e-6


def next_placement(
    queue: Sequence[Job],
    gpus: Sequence[GpuSlots],
    gpu_mem_gib: float,
    max_workers: int,
    now: float = 0.0,
    speed: float = 1.0,
) -> tuple[Job, GpuSlots] | None:
    """The first job in queue order that fits a GPU, on the least loaded GPU it fits.

    If the head of the queue fits nowhere right now, the GPU expected to drain first is
    reserved for it: a later job goes there only if it is expected to end before the drain.
    Times are hours on the clock of ``GpuSlots.ends``.
    """
    reserved: GpuSlots | None = None
    drain = now
    for position, job in enumerate(queue):
        candidates = [
            g
            for g in gpus
            if fits(job, g, gpu_mem_gib, max_workers)
            and (g is not reserved or now + job.est_hours * speed <= drain)
        ]
        if candidates:
            return job, min(candidates, key=lambda g: (len(g.jobs), g.mem_gib, g.index))
        if position == 0 and job.mem_gib <= gpu_mem_gib and gpus:
            reserved = min(
                gpus, key=lambda g: (max(g.ends.values(), default=now), g.mem_gib, g.index)
            )
            drain = max(reserved.ends.values(), default=now)
    return None


def simulate(
    jobs: Sequence[Job], n_gpus: int, gpu_mem_gib: float, max_workers: int, speed: float = 1.0
) -> float:
    """Makespan in hours of ``jobs`` under the runner's placement rule."""
    by_id = {j.id: j for j in jobs}
    queue = sorted((j for j in jobs if j.mem_gib <= gpu_mem_gib), key=lambda j: -j.est_hours)
    gpus = [GpuSlots(i) for i in range(n_gpus)]
    running: list[tuple[float, int, str]] = []  # (end, gpu, job id)
    now = 0.0
    while queue or running:
        while (
            placement := next_placement(queue, gpus, gpu_mem_gib, max_workers, now, speed)
        ) is not None:
            job, gpu = placement
            queue.remove(job)
            end = now + job.est_hours * speed
            gpu.add(job, end)
            heapq.heappush(running, (end, gpu.index, job.id))
        if not running:
            break
        now, index, job_id = heapq.heappop(running)
        gpus[index].remove(by_id[job_id])
    return now


def visible_gpus() -> list[int]:
    out = subprocess.run(  # nosec B603 B607 - fixed nvidia-smi query
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=True,
    )
    return [int(line) for line in out.stdout.split()]


def parse_gpus(spec: str) -> list[int]:
    """'0-3,6' -> [0, 1, 2, 3, 6]."""
    gpus: list[int] = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        gpus.extend(range(int(lo), int(hi or lo) + 1))
    return gpus


def git_sha() -> str:
    out = subprocess.run(  # nosec B603 B607 - fixed git query
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def find_results(job: Job, started: float) -> Path | None:
    """The results.json under the job's --output_dir that records the job's argv."""
    argv = list(job.argv)
    output_dir = (
        Path(argv[argv.index('--output_dir') + 1]) if '--output_dir' in argv else Path('.')
    )
    root = REPO_ROOT / job.cwd / output_dir
    for path in root.glob('*/results.json'):
        if path.stat().st_mtime < started - 1:
            continue
        try:
            recorded = json.loads(path.read_text()).get('argv')
        except (OSError, json.JSONDecodeError):
            continue
        if recorded == argv[1:]:
            return path
    return None


@dataclass
class Running:
    job: Job
    gpu: GpuSlots
    proc: subprocess.Popen[bytes]
    log: IO[bytes]
    started: float
    attempt: int


class Runner:
    def __init__(
        self,
        jobs: Sequence[Job],
        gpus: Sequence[int],
        gpu_mem_gib: float,
        max_workers: int,
        state_dir: Path,
        python: str,
        omp_threads: int,
        retries: int,
        progress_every_s: float,
        poll_s: float = 5.0,
        manifest: Path | None = None,
        machine: str = '',
        speed: float = 1.0,
    ) -> None:
        self.gpus = [GpuSlots(i) for i in gpus]
        self.speed = speed
        self.gpu_mem_gib = gpu_mem_gib
        self.max_workers = max_workers
        self.state_dir = state_dir
        self.python = python
        self.omp_threads = omp_threads
        self.retries = retries
        self.progress_every_s = progress_every_s
        self.poll_s = poll_s
        self.host = socket.gethostname().split('.')[0]
        self.git_sha = git_sha()
        for sub in ('done', 'failed', 'logs'):
            (state_dir / sub).mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.machine = machine
        self.manifest_mtime = manifest.stat().st_mtime if manifest else 0.0
        self.jobs = list(jobs)
        self.skipped = [j for j in jobs if self._is_done(j)]
        self.queue = sorted((j for j in jobs if not self._is_done(j)), key=queue_order)
        self.attempts: dict[str, int] = {}
        self.running: list[Running] = []
        self.failed: list[str] = []
        self.stop = False

    def _is_done(self, job: Job) -> bool:
        return (self.state_dir / 'done' / f'{job.id}.json').exists()

    def _reload(self) -> None:
        """Pick up a changed manifest: new jobs, jobs moved elsewhere, updated estimates.

        Running jobs are never interrupted, even if they were moved to another machine.
        """
        if self.manifest is None or self.manifest.stat().st_mtime == self.manifest_mtime:
            return
        try:
            jobs = jobs_for(load_manifest(self.manifest), self.machine)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            self.log(f'manifest reload failed, keeping the current queue: {exc}')
            return
        self.manifest_mtime = self.manifest.stat().st_mtime
        ids = {j.id for j in jobs}
        busy = {j.id for j in self.queue} | {r.job.id for r in self.running}
        dropped = [j for j in self.queue if j.id not in ids]
        added = [j for j in jobs if j.id not in busy and not self._is_done(j)]
        self.jobs = jobs
        current = {j.id: j for j in jobs}
        kept = [current[j.id] for j in self.queue if j.id in ids]
        self.queue = sorted(kept + added, key=queue_order)
        self.log(
            f'manifest reloaded: {len(added)} jobs added, {len(dropped)} dropped from the queue'
        )

    def log(self, message: str) -> None:
        print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)

    def _write_marker(self, kind: str, job: Job, record: dict[str, Any]) -> None:
        path = self.state_dir / kind / f'{job.id}.json'
        path.write_text(json.dumps(record, indent=2) + '\n')

    def _start(self, job: Job, gpu: GpuSlots) -> None:
        attempt = self.attempts.get(job.id, 0) + 1
        self.attempts[job.id] = attempt
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu.index),
            OMP_NUM_THREADS=str(self.omp_threads),
            PYTHONUNBUFFERED='1',
            PYTHONPATH=f'{REPO_ROOT / "src"}:{env.get("PYTHONPATH", "")}'.rstrip(':'),
        )
        log = (self.state_dir / 'logs' / f'{job.label}.log').open('ab')
        header = f'=== attempt {attempt} on {self.host} GPU {gpu.index} at {self.git_sha[:7]} — {datetime.now()}\n'
        log.write(header.encode())
        log.flush()
        proc = subprocess.Popen(  # nosec B603 - argv from the manifest, no shell
            [self.python, *job.argv],
            cwd=REPO_ROOT / job.cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        gpu.add(job, time.time() / 3600 + job.est_hours * self.speed)
        self.running.append(Running(job, gpu, proc, log, time.time(), attempt))
        self.log(
            f'start GPU {gpu.index} ({gpu.mem_gib:.1f} GiB est) attempt {attempt}: {job.label}'
        )

    def _finish(self, run: Running, code: int) -> None:
        run.log.close()
        run.gpu.remove(run.job)
        wall = time.time() - run.started
        results = find_results(run.job, run.started) if code == 0 else None
        record = {
            'id': run.job.id,
            'label': run.job.label,
            'exit_code': code,
            'host': self.host,
            'gpu': run.gpu.index,
            'git_sha': self.git_sha,
            'attempt': run.attempt,
            'wall_time_s': round(wall, 1),
            'finished': datetime.now().isoformat(timespec='seconds'),
            'results_json': os.path.relpath(results, REPO_ROOT) if results else None,
        }
        if code == 0 and results is not None:
            self._write_marker('done', run.job, record)
            (self.state_dir / 'failed' / f'{run.job.id}.json').unlink(missing_ok=True)
            self.log(f'done ({wall / 3600:.2f} h): {run.job.label}')
            return
        if code == 0:
            record['error'] = 'exit 0 but no matching results.json'
        self._write_marker('failed', run.job, record)
        if self.stop:
            return
        if run.attempt <= self.retries:
            self.log(f'FAILED (exit {code}), retrying alone on a GPU: {run.job.label}')
            self.queue.insert(0, replace(run.job, mem_gib=self.gpu_mem_gib))
            return
        self.failed.append(run.job.label)
        self.log(f'FAILED (exit {code}), giving up: {run.job.label}')

    def _reap(self) -> None:
        for run in list(self.running):
            code = run.proc.poll()
            if code is not None:
                self.running.remove(run)
                self._finish(run, code)

    def _progress(self) -> None:
        per_gpu = {
            str(g.index): {
                'mem_gib_est': round(g.mem_gib, 1),
                'jobs': [r.job.label for r in self.running if r.gpu is g],
            }
            for g in self.gpus
        }
        done = sum(self._is_done(j) for j in self.jobs)
        state = {
            'host': self.host,
            'git_sha': self.git_sha,
            'time': datetime.now().isoformat(timespec='seconds'),
            'total': len(self.jobs),
            'done': done,
            'failed': self.failed,
            'running': len(self.running),
            'queued': len(self.queue),
            'queued_est_hours': round(sum(j.est_hours for j in self.queue), 1),
            'gpus': per_gpu,
        }
        (self.state_dir / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')
        self.log(
            f'progress: {done}/{len(self.jobs)} done, {len(self.failed)} failed, '
            f'{len(self.running)} running, {len(self.queue)} queued ({state["queued_est_hours"]} est h)'
        )

    def _terminate_all(self, signum: int, _frame: object) -> None:
        self.stop = True
        self.log(f'signal {signum}: terminating {len(self.running)} jobs')
        for run in self.running:
            try:
                os.killpg(run.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self) -> int:
        signal.signal(signal.SIGINT, self._terminate_all)
        signal.signal(signal.SIGTERM, self._terminate_all)
        signal.signal(signal.SIGHUP, self._terminate_all)
        unrunnable = [j for j in self.queue if j.mem_gib > self.gpu_mem_gib]
        for job in unrunnable:
            self.queue.remove(job)
            self.failed.append(job.label)
            self.log(
                f'REFUSED ({job.mem_gib:.1f} GiB > {self.gpu_mem_gib:.1f} GiB per GPU): {job.label}'
            )
        self.log(
            f'{self.host} at {self.git_sha[:7]}: {len(self.jobs)} jobs, {len(self.skipped)} already done, '
            f'{len(self.queue)} queued; GPUs {[g.index for g in self.gpus]}, '
            f'{self.gpu_mem_gib:.1f} GiB and {self.max_workers} jobs per GPU'
        )
        last_progress = 0.0
        while True:
            self._reap()
            if self.stop:
                break
            self._reload()
            while (
                placement := next_placement(
                    self.queue,
                    self.gpus,
                    self.gpu_mem_gib,
                    self.max_workers,
                    time.time() / 3600,
                    self.speed,
                )
            ) is not None:
                job, gpu = placement
                self.queue.remove(job)
                self._start(job, gpu)
            if not self.queue and not self.running:
                break
            if time.time() - last_progress >= self.progress_every_s:
                self._progress()
                last_progress = time.time()
            time.sleep(self.poll_s)
        for run in list(self.running):
            run.proc.wait()
            self.running.remove(run)
            self._finish(run, run.proc.returncode)
        self._progress()
        if self.stop:
            self.log('stopped by signal; rerun to resume')
            return 130
        if self.failed:
            self.log(f'{len(self.failed)} failed: {", ".join(self.failed)}')
            return 1
        self.log('all jobs done')
        return 0


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jobs_for(manifest: dict[str, Any], machine: str) -> list[Job]:
    return [Job.from_dict(j) for j in manifest['jobs'] if j['machine'] == machine]


def queue_order(job: Job) -> tuple[float, str]:
    return (-job.est_hours, job.label)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--machine', required=True, help='run the jobs assigned to this machine')
    parser.add_argument('--state_dir', type=Path, required=True)
    parser.add_argument(
        '--gpus', default=None, help="e.g. '0-7'; default: every GPU nvidia-smi lists"
    )
    parser.add_argument(
        '--gpu_mem_gib', type=float, default=None, help='default: the manifest machine spec'
    )
    parser.add_argument(
        '--max_workers_per_gpu', type=int, default=None, help='default: the manifest machine spec'
    )
    parser.add_argument('--omp_threads', type=int, default=8)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--progress_every_s', type=float, default=600.0)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument(
        '--dry_run', action='store_true', help='print the queue and the simulated makespan'
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    spec = manifest['machines'][args.machine]
    gpu_mem_gib = args.gpu_mem_gib if args.gpu_mem_gib is not None else float(spec['gpu_mem_gib'])
    max_workers = (
        args.max_workers_per_gpu
        if args.max_workers_per_gpu is not None
        else int(spec['max_workers_per_gpu'])
    )
    gpus = parse_gpus(args.gpus) if args.gpus else visible_gpus()
    jobs = jobs_for(manifest, args.machine)
    runner = Runner(
        jobs,
        gpus,
        gpu_mem_gib,
        max_workers,
        args.state_dir,
        args.python,
        args.omp_threads,
        args.retries,
        args.progress_every_s,
        manifest=args.manifest,
        machine=args.machine,
        speed=float(spec.get('speed', 1.0)),
    )
    if args.dry_run:
        for job in runner.queue:
            print(f'{job.est_hours:6.2f} h {job.mem_gib:5.1f} GiB  {job.label}')
        makespan = simulate(
            runner.queue, len(gpus), gpu_mem_gib, max_workers, float(spec.get('speed', 1.0))
        )
        print(
            f'{len(runner.queue)} queued, {len(runner.skipped)} done; '
            f'simulated makespan {makespan:.1f} h on {len(gpus)} GPUs'
        )
        return 0
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
