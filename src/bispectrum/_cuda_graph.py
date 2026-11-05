"""CUDA-graph replay of inference forwards, shared by all bispectrum modules.

At small batch sizes a bispectrum forward is a few dozen tiny kernels, so its GPU wall-clock is set
by host-side launch overhead rather than by the GPU. Replaying a captured graph issues them as one
launch. Not part of the public API.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, Self

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CUDAGraphModule(nn.Module):
    """Base class for modules whose ``forward`` is decorated with :func:`cuda_graphed`.

    One graph per module, bound to the shape, dtype and device of the first CUDA input seen
    without autograd; inputs with another signature run eagerly until
    :meth:`reset_cuda_graph_cache`. Moving or casting the module (anything that goes through
    ``nn.Module._apply``) drops the graph, since it addresses the old buffer storage.
    """

    def reset_cuda_graph_cache(self) -> None:
        """Drop the captured CUDA graph so the next inference call re-captures."""
        if hasattr(self, '_cuda_graph_cache'):
            del self._cuda_graph_cache

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        self.reset_cuda_graph_cache()
        return super()._apply(fn, recurse)

    @staticmethod
    def _cuda_graph_key(f: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype, torch.device]:
        """Input signature a captured CUDA graph is valid for."""
        return (tuple(f.shape), f.dtype, f.device)

    def _forward_cuda_graph(
        self, f: torch.Tensor, eager: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor | None:
        """Replay (capturing on first use) ``eager`` as a CUDA graph; None means run eagerly.

        The graph is keyed on the full input signature: ``static_input.copy_`` would otherwise
        silently cast, e.g., a float64 input into a float32 graph.
        """
        key = self._cuda_graph_key(f)
        cache: dict[str, Any] | None = getattr(self, '_cuda_graph_cache', None)
        if cache is not None:
            if cache['key'] != key:
                return None
            cache['static_input'].copy_(f)
            cache['graph'].replay()
            return cache['static_output'].clone()

        rng: torch.Generator | None = None
        rng_backup: torch.Generator | None = None
        try:
            # A normal (non-inference) tensor accepts copy_ under both no_grad and
            # inference_mode, so callers may mix the two.
            with torch.inference_mode(False):
                static_input = torch.empty_like(f)
                static_input.copy_(f)

            # A capture that fails inside torch.cuda.graph leaves the default CUDA generator
            # flagged as capturing, and every later CUDA RNG call raises; restore it on failure.
            rng = torch.cuda.default_generators[torch.cuda.current_device()]
            rng_backup = rng.clone_state()

            # Warm up on a side stream: lazy device caches and cuBLAS/cuFFT plans must exist
            # before capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                eager(static_input)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = eager(static_input)

            self._cuda_graph_cache = {
                'graph': g,
                'static_input': static_input,
                'static_output': static_output,
                'key': key,
            }
            static_input.copy_(f)
            g.replay()
            return static_output.clone()

        except Exception:
            if rng is not None and rng_backup is not None:
                rng.graphsafe_set_state(rng_backup)
            logger.debug(
                'CUDA graph capture failed for %s, running eagerly',
                type(self).__name__,
                exc_info=True,
            )
            self._cuda_graph_cache = {'key': None}
            return None


def cuda_graphed[M: CUDAGraphModule](
    forward: Callable[[M, torch.Tensor], torch.Tensor],
) -> Callable[[M, torch.Tensor], torch.Tensor]:
    """Route CUDA inference calls of ``forward`` through
    :meth:`CUDAGraphModule._forward_cuda_graph`."""

    @functools.wraps(forward)
    def wrapper(self: M, f: torch.Tensor) -> torch.Tensor:
        if f.is_cuda and not torch.is_grad_enabled():
            out = self._forward_cuda_graph(f, functools.partial(forward, self))
            if out is not None:
                return out
        return forward(self, f)

    return wrapper
