"""Influence functions solved with conjugate gradients over HVPs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg
from torch import Tensor, nn

from flightrec.probes.hvp import HessianOperator
from flightrec.utils import flatten_tensors, move_batch


@dataclass
class InfluenceConfig:
    """Configuration for damped inverse-Hessian influence estimates."""

    damping: float = 0.01
    cg_tol: float = 1e-4
    cg_maxiter: int = 100
    hessian_batches: int = 8
    last_layers_only: bool = True


def _xy(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("loaders must yield (inputs, targets[, indices])")
    return batch[0], batch[1]


def _criterion_loss(model: nn.Module, loss_fn: Callable[..., Tensor], batch: Any) -> Tensor:
    inputs, targets = _xy(batch)
    try:
        return loss_fn(model(inputs), targets)
    except TypeError as criterion_error:
        try:
            return loss_fn(model, batch)
        except TypeError:
            raise criterion_error from None


def _selected_params(model: nn.Module, last_only: bool) -> list[Tensor]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not last_only:
        return [param for _, param in named]
    conventional = [
        param
        for name, param in named
        if name.startswith(("layer4.", "fc.", "classifier.", "head.")) or ".layer4." in name
    ]
    return conventional or [param for _, param in named[-2:]]


def _system(
    model: nn.Module,
    loss_fn: Callable[..., Tensor],
    hessian_loader: Iterable[Any],
    cfg: InfluenceConfig,
    device: torch.device,
) -> tuple[list[Tensor], LinearOperator]:
    batches = []
    for batch in hessian_loader:
        batches.append(move_batch(batch, device))
        if len(batches) >= cfg.hessian_batches:
            break
    if not batches:
        raise ValueError("hessian_loader yielded no batches")
    params = _selected_params(model, cfg.last_layers_only)

    def closure() -> Tensor:
        return torch.stack([_criterion_loss(model, loss_fn, batch) for batch in batches]).mean()

    hessian = HessianOperator(model, closure, device, params=params)
    damped = LinearOperator(
        hessian.shape,
        matvec=lambda vector: hessian.matvec(vector) + cfg.damping * vector,
        dtype=np.float64,
    )
    return params, damped


def _gradient(loss: Tensor, params: list[Tensor]) -> np.ndarray:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    flat = flatten_tensors(
        grad if grad is not None else torch.zeros_like(param)
        for grad, param in zip(grads, params, strict=True)
    )
    return flat.detach().cpu().double().numpy()


def _safe_dot(left: np.ndarray, right: np.ndarray, label: str) -> float:
    """Return a finite dot product without relying on platform BLAS status flags."""
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            value = 0.0
            for start in range(0, len(left), 1_000_000):
                stop = min(start + 1_000_000, len(left))
                products = np.multiply(left[start:stop], right[start:stop])
                value += float(np.sum(products, dtype=np.float64))
    except FloatingPointError as error:
        raise FloatingPointError(f"{label} overflowed") from error
    if not np.isfinite(value):
        raise FloatingPointError(f"{label} is non-finite")
    return value


def _candidate_gradients(
    model: nn.Module,
    loss_fn: Callable[..., Tensor],
    loader: Iterable[Any],
    params: list[Tensor],
    device: torch.device,
) -> Iterable[np.ndarray]:
    for original in loader:
        batch = move_batch(original, device)
        inputs, targets = _xy(batch)
        # Reuse one forward graph for the whole microbatch. This matters when thousands of
        # candidates share a large frozen feature extractor; only the selected-parameter
        # backward is repeated per example.
        try:
            logits = model(inputs)
            losses = [
                loss_fn(logits[index : index + 1], targets[index : index + 1])
                for index in range(len(targets))
            ]
        except TypeError:
            losses = [
                _criterion_loss(
                    model,
                    loss_fn,
                    (inputs[index : index + 1], targets[index : index + 1]),
                )
                for index in range(len(targets))
            ]
        for index, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                params,
                allow_unused=True,
                retain_graph=index + 1 < len(losses),
            )
            flat = flatten_tensors(
                grad if grad is not None else torch.zeros_like(param)
                for grad, param in zip(grads, params, strict=True)
            )
            yield flat.detach().cpu().double().numpy()


def self_influence(
    model: nn.Module,
    loss_fn: Callable[..., Tensor],
    candidate_loader: Iterable[Any],
    hessian_loader: Iterable[Any],
    cfg: InfluenceConfig,
    device: str | torch.device,
) -> np.ndarray:
    """Compute ``g.T (H + damping I)^-1 g`` for every candidate.

    This exact variant performs one conjugate-gradient solve per candidate; use
    candidate pre-filtering for large datasets.
    """
    target_device = torch.device(device)
    probe_model = copy.deepcopy(model).to(target_device)
    probe_model.eval()
    params, system = _system(probe_model, loss_fn, hessian_loader, cfg, target_device)
    scores = []
    for gradient in _candidate_gradients(
        probe_model, loss_fn, candidate_loader, params, target_device
    ):
        solution, info = cg(system, gradient, rtol=cfg.cg_tol, atol=0.0, maxiter=cfg.cg_maxiter)
        if info != 0:
            raise RuntimeError(f"conjugate gradient did not converge (info={info})")
        if not np.isfinite(solution).all():
            raise FloatingPointError("conjugate gradient returned a non-finite solution")
        scores.append(_safe_dot(gradient, solution, "self-influence score"))
    return np.asarray(scores, dtype=np.float64)


def influence_on(
    model: nn.Module,
    loss_fn: Callable[..., Tensor],
    test_batch: Any,
    candidate_loader: Iterable[Any],
    hessian_loader: Iterable[Any],
    cfg: InfluenceConfig,
    device: str | torch.device,
) -> np.ndarray:
    """Compute each training candidate's influence on a fixed test batch."""
    target_device = torch.device(device)
    probe_model = copy.deepcopy(model).to(target_device)
    probe_model.eval()
    params, system = _system(probe_model, loss_fn, hessian_loader, cfg, target_device)
    test = move_batch(test_batch, target_device)
    test_gradient = _gradient(_criterion_loss(probe_model, loss_fn, test), params)
    inverse_test, info = cg(
        system, test_gradient, rtol=cfg.cg_tol, atol=0.0, maxiter=cfg.cg_maxiter
    )
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge (info={info})")
    if not np.isfinite(inverse_test).all():
        raise FloatingPointError("conjugate gradient returned a non-finite solution")
    values = [
        -_safe_dot(gradient, inverse_test, "influence score")
        for gradient in _candidate_gradients(
            probe_model, loss_fn, candidate_loader, params, target_device
        )
    ]
    return np.asarray(values, dtype=np.float64)
