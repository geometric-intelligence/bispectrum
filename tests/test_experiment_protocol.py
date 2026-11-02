"""Training protocol shared by the experiment scripts.

Every model in a sweep must get the same training budget, otherwise accuracy differences
reflect when a run stopped rather than the model. The experiments use SGDR
(``CosineAnnealingWarmRestarts``, ``T_0 = 10`` epochs, ``T_mult = 2``); validation metrics
dip after each warm restart, so patience-based early stopping fired at restart-dependent
epochs. The protocol is therefore:

1. Early stopping is off by default (``--patience 0``) and no sweep script turns it on.
2. ``--epochs`` (train.py defaults and every sweep script) is an SGDR cycle end, where the
   learning rate is at its minimum: 10, 30, 70, 150, ...
3. Within a dataset, every sweep script trains every model with one batch size.

``train.py`` is checked statically (AST) because importing it needs the optional
experiment dependencies; the cycle ends are taken from torch's scheduler itself.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / 'experiments'
DATASETS = ('organ3d', 'pcam', 'spherical_mnist')
SWEEP_SCRIPTS = sorted(EXPERIMENTS_DIR.glob('*/run_*.sh'))


def _train_ast(dataset: str) -> ast.Module:
    return ast.parse((EXPERIMENTS_DIR / dataset / 'train.py').read_text())


def _argparse_default(tree: ast.Module, flag: str) -> object:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            for kw in node.keywords:
                if kw.arg == 'default':
                    return ast.literal_eval(kw.value)
            return None
    raise AssertionError(f'{flag} not defined')


def _module_constant(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not defined')


def _scheduler_cycle_ends(t0_epochs: int, t_mult: int, max_epoch: int) -> list[int]:
    """Epochs after which torch's SGDR restarts, stepping per batch as train.py does."""
    steps_per_epoch = 3
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=t0_epochs * steps_per_epoch, T_mult=t_mult
    )
    ends: list[int] = []
    for epoch in range(1, max_epoch + 1):
        for _ in range(steps_per_epoch):
            opt.step()
            sched.step()
        if sched.T_cur == 0:
            ends.append(epoch)
    return ends


CYCLE_ENDS = _scheduler_cycle_ends(10, 2, 400)


def test_scheduler_cycle_ends() -> None:
    assert CYCLE_ENDS == [10, 30, 70, 150, 310]


@pytest.mark.parametrize('dataset', DATASETS)
def test_train_defaults_disable_early_stopping(dataset: str) -> None:
    tree = _train_ast(dataset)
    assert _argparse_default(tree, '--patience') == 0
    assert _argparse_default(tree, '--epochs') in CYCLE_ENDS


@pytest.mark.parametrize('dataset', DATASETS)
def test_train_scheduler_uses_protocol_constants(dataset: str) -> None:
    """The scheduler is built from ``SGDR_T0_EPOCHS`` / ``SGDR_T_MULT`` = (10, 2)."""
    tree = _train_ast(dataset)
    assert _module_constant(tree, 'SGDR_T0_EPOCHS') == 10
    assert _module_constant(tree, 'SGDR_T_MULT') == 2
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, 'attr', getattr(node.func, 'id', None))
        == 'CosineAnnealingWarmRestarts'
    ]
    assert len(calls) == 1
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert kwargs == {'T_0': 'SGDR_T0_EPOCHS * steps_per_epoch', 'T_mult': 'SGDR_T_MULT'}


def _script_epochs(text: str) -> list[int]:
    epochs = [int(v) for v in re.findall(r'--epochs\s+(\d+)', text)]
    for body in re.findall(r'epochs_for\(\)\s*\{(.*?)\n\}', text, flags=re.S):
        epochs += [int(v) for v in re.findall(r'echo\s+(\d+)', body)]
    return epochs


@pytest.mark.parametrize('script', SWEEP_SCRIPTS, ids=lambda p: f'{p.parent.name}/{p.name}')
def test_sweep_script_protocol(script: Path) -> None:
    text = script.read_text()
    assert '--patience' not in text, 'sweep scripts must not enable early stopping'
    epochs = _script_epochs(text)
    assert epochs, 'sweep scripts must pin --epochs explicitly'
    assert set(epochs) <= set(CYCLE_ENDS), f'--epochs {epochs} not all SGDR cycle ends'
    assert not re.search(r'--batch_size\s+"?\$', text), 'batch size must not vary per model'


@pytest.mark.parametrize('dataset', DATASETS)
def test_one_batch_size_per_dataset(dataset: str) -> None:
    sizes = {
        int(v)
        for script in SWEEP_SCRIPTS
        if script.parent.name == dataset
        for v in re.findall(r'--batch_size\s+(\d+)', script.read_text())
    }
    assert len(sizes) <= 1, f'{dataset} sweeps use batch sizes {sorted(sizes)}'


def _load_grid_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip('matplotlib')
    path = EXPERIMENTS_DIR / 'make_grid_figure.py'
    spec = importlib.util.spec_from_file_location('_exp_make_grid_figure', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses resolve their module
    spec.loader.exec_module(module)
    return module


def test_stale_reason_rejects_early_stopped_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _load_grid_module(monkeypatch)
    current = {
        'dataset': 'organ3d',
        'train_size': 100,
        'min_epoch_samples': 10_000,
        'bispectrum_version': '0.4.0',
        'early_stopping_patience': 0,
    }
    assert grid._stale_reason(current) is None
    assert grid._stale_reason({**current, 'early_stopping_patience': 15}) == 'early-stopped'
    legacy = {k: v for k, v in current.items() if k != 'early_stopping_patience'}
    assert grid._stale_reason(legacy) == 'early-stopped'


def test_stale_reason_rejects_pre_head_fix_bispectrum_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _load_grid_module(monkeypatch)
    head = {'nonlin': 'post', 'power': False, 'scale': 'log'}
    for dataset in ('pcam', 'organ3d'):
        run = {
            'dataset': dataset,
            'model': 'bispectrum',
            'train_size': None,
            'bispectrum_version': '0.4.0',
            'early_stopping_patience': 0,
        }
        assert grid._stale_reason(run) == 'pre-head-fix'
        assert grid._stale_reason({**run, 'bispectrum_head': head}) is None
        assert grid._stale_reason({**run, 'model': 'max_pool'}) is None
    smnist = {
        'dataset': 'spherical_mnist',
        'model': 'bispectrum',
        'train_size': None,
        'bispectrum_version': '0.4.0',
        'early_stopping_patience': 0,
    }
    assert grid._stale_reason(smnist) is None
