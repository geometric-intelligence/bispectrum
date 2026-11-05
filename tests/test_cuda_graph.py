"""CUDA-graph replay shared by every module (bispectrum._cuda_graph)."""

import functools
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from bispectrum import CnonCn, DnonDn, OctaonOcta, SO2onDisk, SO2onS1, SO3onS2, TorusOnTorus
from bispectrum._cuda_graph import CUDAGraphModule, cuda_graphed

CASES: dict[str, tuple[Callable[[], CUDAGraphModule], tuple[int, ...]]] = {
    'CnonCn': (lambda: CnonCn(n=16), (16,)),
    'SO2onS1': (lambda: SO2onS1(n=12), (12,)),
    'TorusOnTorus': (lambda: TorusOnTorus(ns=(6, 8)), (6, 8)),
    'DnonDn': (lambda: DnonDn(n=6), (12,)),
    'SO2onDisk': (lambda: SO2onDisk(L=8), (8, 8)),
    'SO3onS2': (lambda: SO3onS2(lmax=6, nlat=16, nlon=32), (16, 32)),
    'OctaonOcta': (lambda: OctaonOcta(), (24,)),
}

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _eager(module: CUDAGraphModule) -> Callable[[torch.Tensor], torch.Tensor]:
    return functools.partial(type(module).forward.__wrapped__, module)


def _mock_cuda() -> tuple[MagicMock, list]:
    graph = MagicMock()
    patches = [
        patch('torch.cuda.Stream', return_value=MagicMock()),
        patch('torch.cuda.current_stream', return_value=MagicMock()),
        patch('torch.cuda.stream', return_value=MagicMock()),
        patch('torch.cuda.CUDAGraph', return_value=graph),
        patch('torch.cuda.graph', return_value=MagicMock()),
        patch('torch.cuda.current_device', return_value=0),
        patch('torch.cuda.default_generators', (MagicMock(),)),
    ]
    return graph, patches


@pytest.mark.parametrize('cast', ['double', 'float', 'to_cpu'])
def test_apply_drops_captured_graph(cast: str):
    module = CnonCn(n=8)
    module._cuda_graph_cache = {'key': None}
    {'double': module.double, 'float': module.float, 'to_cpu': module.cpu}[cast]()
    assert not hasattr(module, '_cuda_graph_cache')


def test_static_input_accepts_no_grad_after_inference_mode_capture():
    """Capturing under inference_mode must not make the replay buffer an inference tensor."""
    module = CnonCn(n=8)
    f = torch.randn(2, 8)
    graph, patches = _mock_cuda()
    for p in patches:
        p.start()
    try:
        with torch.inference_mode():
            module._forward_cuda_graph(f, _eager(module))
        assert not module._cuda_graph_cache['static_input'].is_inference()
        with torch.no_grad():
            out = module._forward_cuda_graph(f, _eager(module))
    finally:
        for p in patches:
            p.stop()
    assert out is not None
    assert graph.replay.call_count == 2


@cuda
@pytest.mark.parametrize('name', list(CASES))
def test_graph_replay_matches_eager(name: str):
    make, shape = CASES[name]
    module = make().cuda()
    gen = torch.Generator(device='cuda').manual_seed(0)
    f1 = torch.randn(4, *shape, device='cuda', generator=gen)
    f2 = torch.randn(4, *shape, device='cuda', generator=gen)
    with torch.inference_mode():
        out1 = module(f1)
        assert module._cuda_graph_cache['key'] is not None, 'capture fell back to eager'
        out2 = module(f2)
        ref1, ref2 = _eager(module)(f1), _eager(module)(f2)
    torch.testing.assert_close(out1, ref1, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out2, ref2, rtol=1e-4, atol=1e-5)
    assert out2.data_ptr() != module._cuda_graph_cache['static_output'].data_ptr()


@cuda
@pytest.mark.parametrize('name', list(CASES))
def test_other_batch_size_and_autograd_run_eagerly(name: str):
    make, shape = CASES[name]
    module = make().cuda()
    with torch.no_grad():
        module(torch.randn(4, *shape, device='cuda'))
        f8 = torch.randn(8, *shape, device='cuda')
        out8 = module(f8)
        torch.testing.assert_close(out8, _eager(module)(f8), rtol=1e-4, atol=1e-5)
    f = torch.randn(4, *shape, device='cuda', requires_grad=True)
    module(f).abs().sum().backward()
    assert f.grad is not None


class _Uncapturable(CUDAGraphModule):
    @cuda_graphed
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Boolean-mask indexing syncs with the host, which is illegal while capturing.
        return 2 * f[:, f[0] > -1e9]


@cuda
def test_failed_capture_falls_back_and_keeps_cuda_rng_usable():
    module = _Uncapturable()
    f = torch.randn(4, 8, device='cuda')
    with torch.no_grad():
        out = module(f)
    assert module._cuda_graph_cache == {'key': None}
    torch.testing.assert_close(out, 2 * f)
    assert torch.isfinite(torch.randn(16, device='cuda')).all()


@cuda
def test_reallocated_buffers_recapture():
    """After .cpu().cuda() the old graph would read freed buffer storage."""
    module = SO3onS2(lmax=6, nlat=16, nlon=32).cuda()
    f = torch.randn(4, 16, 32, device='cuda')
    with torch.no_grad():
        ref = module(f)
        module.cpu().cuda()
        module._apply(lambda t: t.clone())
        junk = [torch.full((1 << 16,), float('nan'), device='cuda') for _ in range(64)]
        out = module(f)
    del junk
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
