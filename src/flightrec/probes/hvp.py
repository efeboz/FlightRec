"""Hessian-vector products and SciPy interoperability."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from torch import Tensor, nn

from flightrec.utils import flatten_tensors, unflatten_vector

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibility path for older supported PyTorch
    SDPBackend = None
    sdpa_kernel = None


@contextmanager
def _math_attention() -> Iterator[None]:
    """Force the attention backend with CPU double-backward support."""
    if sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    else:  # pragma: no cover - exercised only on older supported PyTorch
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            yield


def hvp(loss_fn: Callable[[], Tensor], params: list[Tensor], vector: Tensor) -> Tensor:
    """Compute a Hessian-vector product by double backward.

    Parameters
    ----------
    loss_fn:
        Zero-argument closure that recomputes a scalar loss.
    params:
        Tensors with respect to which the Hessian is taken.
    vector:
        Flat vector with one element per parameter element.
    """
    if vector.numel() != sum(param.numel() for param in params):
        raise ValueError("vector size does not match parameters")
    with _math_attention():
        loss = loss_fn()
        first = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        differentiable = [
            grad if grad is not None else torch.zeros_like(param)
            for grad, param in zip(first, params, strict=True)
        ]
        flat_first = flatten_tensors(differentiable)
        if not flat_first.requires_grad:
            return torch.zeros_like(vector)
        second = torch.autograd.grad(
            flat_first,
            params,
            grad_outputs=vector.to(flat_first),
            allow_unused=True,
        )
    return flatten_tensors(
        grad if grad is not None else torch.zeros_like(param)
        for grad, param in zip(second, params, strict=True)
    ).detach()


class HessianOperator(LinearOperator):
    """Float64 SciPy linear operator over a PyTorch model Hessian.

    The model and closure must already operate on ``device``. ``param_filter``
    may select a subset of named parameters, which is useful for last-layer
    influence approximations.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_closure: Callable[[], Tensor],
        device: str | torch.device,
        param_filter: Callable[[str, Tensor], bool] | None = None,
        params: Iterable[Tensor] | None = None,
    ) -> None:
        self.model = model
        self.loss_closure = loss_closure
        self.device = torch.device(device)
        if params is not None:
            self.params = [param for param in params if param.requires_grad]
        else:
            self.params = [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and (param_filter is None or param_filter(name, param))
            ]
        if not self.params:
            raise ValueError("HessianOperator has no selected parameters")
        size = sum(param.numel() for param in self.params)
        super().__init__(dtype=np.dtype(np.float64), shape=(size, size))

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        reference = self.params[0]
        tensor = torch.as_tensor(np.asarray(vector), dtype=reference.dtype, device=self.device)
        result = hvp(self.loss_closure, self.params, tensor)
        return result.detach().cpu().double().numpy()

    def torch_vector(self, vector: np.ndarray) -> list[Tensor]:
        """Convert a NumPy flat vector to selected parameter-shaped tensors."""
        reference = self.params[0]
        tensor = torch.as_tensor(vector, dtype=reference.dtype, device=self.device)
        return unflatten_vector(tensor, self.params)
