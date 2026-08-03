"""Independent numerical and graph-structure gradient checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class GradReport:
    """Summary returned by FlightRec's gradient sanity checks."""

    max_abs_err_fd: float
    max_abs_err_cs: float | None
    unreachable_params: list[str]
    nonfinite_grads: list[str]
    passed: bool


def _call(fn: Callable[..., Tensor], params: dict[str, Tensor]) -> Tensor:
    try:
        return fn(params)
    except TypeError as mapping_error:
        try:
            return fn(**params)
        except TypeError:
            raise mapping_error from None


def check_gradients(
    fn: Callable[..., Tensor],
    params: dict[str, Tensor],
    eps_fd: float = 1e-6,
    h_cs: float = 1e-20,
) -> GradReport:
    """Compare autograd with central differences and complex-step derivatives.

    ``fn`` may accept the parameter dictionary as one argument or accept its
    entries as keyword arguments. At most 200 deterministically selected flat
    coordinates are checked in float64.
    """
    real = {
        name: value.detach().to(dtype=torch.float64).clone().requires_grad_(True)
        for name, value in params.items()
    }
    output = _call(fn, real)
    analytical_parts = torch.autograd.grad(output, list(real.values()), allow_unused=True)
    analytical = np.concatenate(
        [
            (grad if grad is not None else torch.zeros_like(value)).detach().cpu().numpy().ravel()
            for grad, value in zip(analytical_parts, real.values(), strict=True)
        ]
    )
    total = analytical.size
    coordinates = np.linspace(0, total - 1, min(200, total), dtype=np.int64)
    offsets = np.cumsum([0, *[value.numel() for value in real.values()]])

    def locate(flat_index: int) -> tuple[str, int]:
        tensor_index = int(np.searchsorted(offsets[1:], flat_index, side="right"))
        return list(real)[tensor_index], int(flat_index - offsets[tensor_index])

    fd_errors = []
    for coordinate in coordinates:
        name, local = locate(int(coordinate))
        plus = {key: value.detach().clone() for key, value in real.items()}
        minus = {key: value.detach().clone() for key, value in real.items()}
        plus[name].view(-1)[local] += eps_fd
        minus[name].view(-1)[local] -= eps_fd
        estimate = float((_call(fn, plus) - _call(fn, minus)).item()) / (2.0 * eps_fd)
        fd_errors.append(abs(estimate - analytical[coordinate]))
    max_fd = float(max(fd_errors, default=0.0))

    max_cs: float | None
    try:
        cs_errors = []
        complex_base = {
            name: value.detach().to(dtype=torch.complex128).clone() for name, value in real.items()
        }
        for coordinate in coordinates:
            name, local = locate(int(coordinate))
            changed = {key: value.clone() for key, value in complex_base.items()}
            changed[name].view(-1)[local] += 1j * h_cs
            estimate = float(_call(fn, changed).imag.item() / h_cs)
            cs_errors.append(abs(estimate - analytical[coordinate]))
        max_cs = float(max(cs_errors, default=0.0))
    except (RuntimeError, TypeError, ValueError, NotImplementedError):
        max_cs = None

    nonfinite = [
        name
        for name, grad in zip(real, analytical_parts, strict=True)
        if grad is not None and not bool(torch.isfinite(grad).all())
    ]
    passed = max_fd <= max(1e-5, eps_fd * 100) and (max_cs is None or max_cs <= 1e-10)
    passed = passed and not nonfinite
    return GradReport(max_fd, max_cs, [], nonfinite, passed)


def check_model_graph(model: nn.Module, loss: Tensor) -> GradReport:
    """Report trainable parameters unreachable from ``loss.grad_fn``."""
    reached: set[int] = set()
    pending = [loss.grad_fn] if loss.grad_fn is not None else []
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        variable = getattr(node, "variable", None)
        if isinstance(variable, Tensor):
            reached.add(id(variable))
        pending.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    unreachable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and id(param) not in reached
    ]
    nonfinite = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not bool(torch.isfinite(param.grad).all())
    ]
    return GradReport(0.0, None, unreachable, nonfinite, not unreachable and not nonfinite)
