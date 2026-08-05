"""Explicit, low-overhead training recorder."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from flightrec.probes.curvature import lanczos_ritz_spectrum, lanczos_spectrum
from flightrec.probes.hvp import HessianOperator
from flightrec.storage import RunWriter
from flightrec.utils import move_batch


def _total_norm(tensors: list[Tensor], device: torch.device) -> Tensor:
    """Use the public fused norm API when available, with a PyTorch 2.2 fallback."""
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is not None:
        return get_total_norm(tensors, norm_type=2.0)
    if not tensors:  # pragma: no cover - the public helper exists in current PyTorch
        return torch.zeros((), device=device)
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(tensor, ord=2) for tensor in tensors]), ord=2
    )


class FlightRecorder:
    """Record scalar, per-example, and optional Hessian data from a training loop.

    Call :meth:`record_step` after ``loss.backward()`` and before
    ``optimizer.step()`` so gradient statistics describe the current batch.
    """

    def __init__(
        self,
        model: nn.Module,
        run_dir: str | Path,
        num_samples: int | None = None,
        spectrum_every: int | None = None,
        spectrum_k: int = 10,
        spectrum_lanczos_steps: int | None = None,
        probe_device: str | torch.device = "cpu",
        probe_loss_fn: Callable[[nn.Module, object], Tensor] | None = None,
        probe_batch: object | None = None,
    ) -> None:
        if spectrum_every is not None and (probe_loss_fn is None or probe_batch is None):
            raise ValueError("spectrum probes require probe_loss_fn and probe_batch")
        if spectrum_lanczos_steps is not None and spectrum_lanczos_steps < 2 * spectrum_k:
            raise ValueError("spectrum_lanczos_steps must be at least 2 * spectrum_k")
        self.model = model
        self.num_samples = num_samples
        self.spectrum_every = spectrum_every
        self.spectrum_k = spectrum_k
        self.spectrum_lanczos_steps = spectrum_lanczos_steps
        self.probe_device = torch.device(probe_device)
        self.probe_loss_fn = probe_loss_fn
        self.probe_batch = probe_batch
        config = {
            "spectrum_every": spectrum_every,
            "spectrum_k": spectrum_k,
            "spectrum_lanczos_steps": spectrum_lanczos_steps,
            "probe_device": str(probe_device),
        }
        self.writer = RunWriter(run_dir, num_samples, config)
        self.step = 0
        self.epoch = 0
        self.started = time.perf_counter()
        self.closed = False
        self._epoch_has_steps = False
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        if self.num_samples is not None:
            self._correct = np.full(self.num_samples, 255, dtype=np.uint8)
            self._margin = np.full(self.num_samples, np.nan, dtype=np.float16)

    def record_step(
        self,
        *,
        loss: Tensor,
        logits: Tensor | None = None,
        targets: Tensor | None = None,
        sample_indices: Tensor | None = None,
        lr: float | None = None,
    ) -> None:
        """Record one post-backward, pre-optimizer training step."""
        if self.closed:
            raise RuntimeError("recorder is closed")
        with torch.no_grad():
            parameters = list(self.model.parameters())
            gradients = [
                param.grad.detach().float() for param in parameters if param.grad is not None
            ]
            detached = [param.detach().float() for param in parameters]
            if detached:
                device = detached[0].device
                grad_norm = _total_norm(gradients, device)
                param_norm = _total_norm(detached, device)
                grad_norm, param_norm = torch.stack((grad_norm, param_norm)).tolist()
            else:
                grad_norm = param_norm = 0.0
            record: dict[str, Any] = {
                "kind": "step",
                "step": self.step,
                "epoch": self.epoch,
                "wall_time": time.perf_counter() - self.started,
                "loss": float(loss.detach().item()),
                "grad_norm": grad_norm,
                "param_norm": param_norm,
            }
            if lr is not None:
                record["lr"] = float(lr)
            self.writer.append_scalar(record)
            if self.num_samples is not None and all(
                value is not None for value in (logits, targets, sample_indices)
            ):
                assert logits is not None and targets is not None and sample_indices is not None
                detached_logits = logits.detach()
                detached_targets = targets.detach().long()
                predictions = detached_logits.argmax(dim=-1)
                true_logits = detached_logits.gather(1, detached_targets[:, None]).squeeze(1)
                masked = detached_logits.clone()
                masked.scatter_(1, detached_targets[:, None], -torch.inf)
                margins = true_logits - masked.max(dim=1).values
                indices_np = sample_indices.detach().cpu().numpy().astype(np.int64)
                self._correct[indices_np] = (
                    (predictions == detached_targets).cpu().numpy().astype(np.uint8)
                )
                self._margin[indices_np] = margins.float().cpu().numpy().astype(np.float16)
        self._epoch_has_steps = True
        self.step += 1
        if self.spectrum_every is not None and self.step % self.spectrum_every == 0:
            self._record_spectrum()

    def record_eval(self, **metrics: float) -> None:
        """Record arbitrary scalar evaluation metrics at the current step."""
        record: dict[str, Any] = {
            "kind": "eval",
            "step": self.step,
            "epoch": self.epoch,
            "wall_time": time.perf_counter() - self.started,
        }
        record.update({name: float(value) for name, value in metrics.items()})
        self.writer.append_scalar(record)

    def epoch_end(self) -> None:
        """Flush the current epoch's per-example state and scalar stream."""
        if self.num_samples is not None:
            self.writer.append_epoch(self._correct, self._margin)
        else:
            self.writer.flush()
        self.epoch += 1
        self._epoch_has_steps = False
        self._reset_epoch()

    def _record_spectrum(self) -> None:
        assert self.probe_loss_fn is not None
        probe_model = copy.deepcopy(self.model).to(self.probe_device)
        # A matrix-free eigensolver assumes every matvec represents the same operator.
        # Evaluation mode freezes dropout and BatchNorm buffers on the isolated copy.
        probe_model.eval()
        batch = move_batch(self.probe_batch, self.probe_device)

        def closure() -> Tensor:
            assert self.probe_loss_fn is not None
            return self.probe_loss_fn(probe_model, batch)

        operator = HessianOperator(probe_model, closure, self.probe_device)
        if self.spectrum_lanczos_steps is None:
            low, high = lanczos_spectrum(operator, self.spectrum_k)
        else:
            low, high = lanczos_ritz_spectrum(
                operator,
                self.spectrum_k,
                self.spectrum_lanczos_steps,
                seed=self.step,
            )
        self.writer.append_spectrum(self.step, low, high)

    def close(self) -> None:
        """Write final metadata and close the recorder; idempotent."""
        if self.closed:
            return
        if self.num_samples is not None and self._epoch_has_steps:
            self.epoch_end()
        self.writer.close(self.epoch)
        self.closed = True

    def __enter__(self) -> FlightRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
