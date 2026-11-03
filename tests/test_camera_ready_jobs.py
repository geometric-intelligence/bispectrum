"""Camera-ready job manifest and GPU job runner (rebuttal/camera_ready_manifest.py,
rebuttal/gpu_job_runner.py).

The manifest is the captured ``train.py`` calls of the sweep scripts, so these tests run the
real scripts in capture mode (in a copy of the repository layout with an empty venv) and
check the protocol of every job: the fixed epoch budget at an SGDR cycle end, no early
stopping, one batch size per dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'rebuttal'))

import camera_ready_manifest as crm  # noqa: E402
from gpu_job_runner import GpuSlots, Job, Runner, next_placement, simulate  # noqa: E402

PIPELINES = {'pcam': 90, 'pcam_12500': 21, 'pcam_full': 21, 'pcam_matched': 111, 'organ3d': 280}
SCRIPTS = (
    'rebuttal/run_camera_ready.sh',
    'experiments/pcam/run_data_pareto_sweep.sh',
    'experiments/pcam/run_matched_sweep.sh',
    'experiments/organ3d/run_dataeff_multiseed.sh',
    'experiments/organ3d/run_wider_multiseed.sh',
)
SGDR_CYCLE_ENDS = {10, 30, 70, 150}
BATCH_SIZE = {'pcam': '64', 'organ3d': '16'}


@pytest.fixture(scope='module')
def raw_manifest(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if shutil.which('bash') is None:
        pytest.skip('bash not available')
    root = tmp_path_factory.mktemp('repo')
    for script in SCRIPTS:
        (root / script).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / script, root / script)
    (root / '.venv' / 'bin').mkdir(parents=True)
    (root / '.venv' / 'bin' / 'activate').write_text('')
    capture = root / 'capture'
    subprocess.run(  # nosec B603 B607
        ['bash', str(root / 'rebuttal/run_camera_ready.sh'), 'capture', str(capture), *PIPELINES],
        check=True,
        capture_output=True,
        text=True,
    )
    return crm.build(capture)


def test_capture_job_counts(raw_manifest: dict[str, Any]) -> None:
    assert raw_manifest['pipelines'] == PIPELINES
    assert len({j['id'] for j in raw_manifest['jobs']}) == sum(PIPELINES.values())


def test_captured_jobs_follow_the_protocol(raw_manifest: dict[str, Any]) -> None:
    for job in raw_manifest['jobs']:
        argv = job['argv']
        assert argv[0] == 'train.py', job['label']
        assert '--patience' not in argv, job['label']
        assert '--bispectrum_head' not in argv, job['label']
        epochs = int(crm.flag(argv, '--epochs')[0])
        assert epochs in SGDR_CYCLE_ENDS, job['label']
        assert crm.flag(argv, '--batch_size') == [BATCH_SIZE[job['dataset']]], job['label']
        if job['pipeline'] == 'pcam_full':
            assert epochs == 10 and '--train_size' not in argv
        if job['dataset'] == 'organ3d':
            assert epochs == 70


def _memory_records(raw: dict[str, Any], big: dict[str, float | None]) -> list[dict[str, Any]]:
    records = []
    for config in sorted({j['config'] for j in raw['jobs']}):
        peak = big.get(config, 8.0)
        status = 'ok' if peak is not None else 'fail'
        records.append(
            {
                'config': config,
                'status': status,
                'max_memory_reserved_bytes': int((peak or 0) * 2**30),
            }
        )
    return records


def test_assignment_keeps_big_jobs_off_40gb_gpus(raw_manifest: dict[str, Any]) -> None:
    big = {'pcam_fourier_elu_gr12_C_bs64': None, 'pcam_norm_gr12_C_bs64': 36.5}
    peaks = crm.memory_table(_memory_records(raw_manifest, big))
    manifest = crm.build_manifest(raw_manifest, peaks, crm.CostModel([]), dict(crm.MACHINES))
    machines = manifest['machines']
    assert len(manifest['jobs']) == sum(PIPELINES.values())
    for job in manifest['jobs']:
        assert job['mem_gib'] <= machines[job['machine']]['gpu_mem_gib'], job['label']
        if job['config'] in big:
            assert job['machine'] != 'l8a100', job['label']
        if job['machine'] == 'l8a100':
            assert job['peak_gib'] is not None and job['peak_gib'] <= crm.SMALL_GPU_MAX_PEAK_GIB


def _job(name: str, mem_gib: float, est_hours: float = 1.0, **kwargs: Any) -> Job:
    return Job(
        id=name,
        label=name,
        cwd='.',
        argv=('train.py',),
        mem_gib=mem_gib,
        est_hours=est_hours,
        **kwargs,
    )


def test_two_small_jobs_share_a_40gb_gpu() -> None:
    gpu = GpuSlots(0)
    queue = [_job('a', 18.0), _job('b', 18.0)]
    for expected in queue:
        job, target = next_placement(queue, [gpu], 39.0, 2)
        assert job == expected and target is gpu
        queue.remove(job)
        gpu.add(job, job.est_hours)
    assert simulate([_job('a', 18.0), _job('b', 18.0)], 1, 39.0, 2) == pytest.approx(1.0)


def test_job_too_big_for_the_gpu_is_refused(tmp_path: Path) -> None:
    assert next_placement([_job('big', 48.0)], [GpuSlots(0)], 39.0, 2) is None
    runner = Runner([_job('big', 48.0)], [0], 39.0, 2, tmp_path, sys.executable, 1, 0, 3600.0)
    assert runner.run() == 1
    assert not list((tmp_path / 'done').iterdir())


def test_head_of_queue_reserves_a_gpu() -> None:
    """A long job that needs a whole GPU is not starved by short jobs backfilling."""
    gpus = [GpuSlots(0, ['x'], 10.0), GpuSlots(1, ['y'], 20.0)]
    queue = [_job('big', 30.0, 5.0), _job('small', 8.0)]
    job, gpu = next_placement(queue, gpus, 39.0, 2)
    assert (job.id, gpu.index) == ('small', 1)


def test_backfill_on_the_reserved_gpu_only_before_it_drains() -> None:
    gpus = [GpuSlots(0, ['x'], 30.0, {'x': 5.0}), GpuSlots(1, ['y'], 30.0, {'y': 6.0})]
    big = _job('big', 30.0, 9.0)
    job, gpu = next_placement([big, _job('short', 8.0, 1.0)], gpus, 39.0, 2)
    assert (job.id, gpu.index) == ('short', 0)
    job, gpu = next_placement([big, _job('long', 8.0, 7.0)], gpus, 39.0, 2)
    assert (job.id, gpu.index) == ('long', 1)
    big_first = [_job('big', 30.0, 9.0), _job('a', 8.0, 4.0), _job('b', 8.0, 4.0)]
    assert simulate(big_first, 1, 39.0, 2) == pytest.approx(9.0)


FAKE_TRAIN = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    out = Path(args[args.index('--output_dir') + 1]) / f'run_{args[args.index("--seed") + 1]}'
    if '--fail' in args:
        sys.exit(3)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps({'argv': args}))
    """
)


def _fake_jobs(tmp_path: Path, seeds: list[str], fail: bool = False) -> list[Job]:
    (tmp_path / 'train.py').write_text(FAKE_TRAIN)
    cwd = os.path.relpath(tmp_path, REPO_ROOT)
    jobs = []
    for seed in seeds:
        argv = ('train.py', '--output_dir', 'out', '--seed', seed, *(('--fail',) if fail else ()))
        jobs.append(
            Job(
                id=f'id{seed}', label=f'job{seed}', cwd=cwd, argv=argv, mem_gib=10.0, est_hours=1.0
            )
        )
    return jobs


def _runner(jobs: list[Job], state: Path, retries: int = 0) -> Runner:
    return Runner(jobs, [0, 1], 39.0, 2, state, sys.executable, 1, retries, 3600.0, poll_s=0.05)


def test_runner_writes_done_markers_and_resumes(tmp_path: Path) -> None:
    jobs = _fake_jobs(tmp_path, ['1', '2', '3'])
    state = tmp_path / 'state'
    assert _runner(jobs, state).run() == 0
    for job in jobs:
        marker = json.loads((state / 'done' / f'{job.id}.json').read_text())
        assert marker['exit_code'] == 0
        assert (REPO_ROOT / marker['results_json']).is_file()
        assert (state / 'logs' / f'{job.label}.log').is_file()
    rerun = _runner(jobs, state)
    assert rerun.queue == [] and len(rerun.skipped) == 3


def test_runner_retries_then_reports_failure(tmp_path: Path) -> None:
    jobs = _fake_jobs(tmp_path, ['7'], fail=True)
    state = tmp_path / 'state'
    runner = _runner(jobs, state, retries=1)
    assert runner.run() == 1
    assert runner.attempts == {'id7': 2}
    assert json.loads((state / 'failed' / 'id7.json').read_text())['exit_code'] == 3


def test_runner_picks_up_a_reassigned_manifest(tmp_path: Path) -> None:
    jobs = _fake_jobs(tmp_path, ['1', '2'])
    manifest = tmp_path / 'manifest.json'

    def write(machines: list[str], mtime: int) -> None:
        records = [{**j.to_dict(), 'machine': m} for j, m in zip(jobs, machines, strict=True)]
        manifest.write_text(json.dumps({'jobs': records}))
        os.utime(manifest, (mtime, mtime))

    write(['a', 'b'], 1_000_000)
    runner = Runner(
        [jobs[0]],
        [0],
        39.0,
        2,
        tmp_path / 'state',
        sys.executable,
        1,
        0,
        3600.0,
        manifest=manifest,
        machine='a',
    )
    write(['b', 'a'], 2_000_000)
    runner._reload()
    assert [j.id for j in runner.queue] == ['id2']
    jobs[1] = replace(jobs[1], mem_gib=30.0)
    write(['b', 'a'], 3_000_000)
    runner._reload()
    assert [(j.id, j.mem_gib) for j in runner.queue] == [('id2', 30.0)]
