"""Automatic training-phase segmentation and heuristic labels."""

from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from scipy.interpolate import interp1d

from flightrec.storage import RunData


@dataclass
class Phase:
    """One contiguous phase on the training step axis."""

    start_step: int
    end_step: int
    label: str


@dataclass
class PhaseResult:
    """Change points, labeled segments, and the signals used."""

    breakpoints: list[int]
    phases: list[Phase]
    signals_used: list[str]


def _ema(values: np.ndarray, halflife: float) -> np.ndarray:
    alpha = 1.0 - np.exp(np.log(0.5) / max(halflife, 1.0))
    result = values.copy()
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _interpolate(source_steps: np.ndarray, values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_steps) & np.isfinite(values)
    if valid.sum() == 0:
        return np.full(len(steps), np.nan)
    if valid.sum() == 1:
        return np.full(len(steps), values[valid][0])
    order = np.argsort(source_steps[valid])
    x = source_steps[valid][order]
    y = values[valid][order]
    interpolator = interp1d(
        x, y, kind="linear", bounds_error=False, fill_value=(float(y[0]), float(y[-1]))
    )
    return np.asarray(interpolator(steps), dtype=float)


def detect_phases(run: RunData, penalty: float | None = None) -> PhaseResult:
    """Detect RBF-kernel change points in available, normalized traces."""
    scalars = run.scalars
    if "step" not in scalars:
        return PhaseResult([], [], [])
    kinds = scalars.get("kind", np.full(len(scalars["step"]), "step"))
    step_mask = kinds == "step"
    steps = scalars["step"][step_mask].astype(np.int64)
    if not len(steps):
        return PhaseResult([], [], [])
    unique, positions = np.unique(steps, return_index=True)
    steps = unique
    columns: list[np.ndarray] = []
    names: list[str] = []

    for field, label, logarithm in (
        ("loss", "train_loss", True),
        ("grad_norm", "grad_norm", True),
        ("param_norm", "param_norm", True),
    ):
        if field in scalars:
            raw = scalars[field][step_mask][positions].astype(float)
            if np.isfinite(raw).sum() >= 2:
                filled = _interpolate(steps, raw, steps)
                if logarithm:
                    filled = np.log(np.maximum(filled, 1e-12))
                columns.append(_ema(filled, 0.02 * len(steps)))
                names.append(label)

    if run.spectrum_steps is not None and run.eigs_high is not None and len(run.spectrum_steps):
        top = np.max(run.eigs_high, axis=1)
        columns.append(_interpolate(run.spectrum_steps, top, steps))
        names.append("top_eigenvalue")

    if "test_acc" in scalars:
        eval_mask = np.isfinite(scalars["test_acc"].astype(float))
        if eval_mask.sum() >= 2:
            columns.append(
                _interpolate(
                    scalars["step"][eval_mask].astype(float),
                    scalars["test_acc"][eval_mask].astype(float),
                    steps,
                )
            )
            names.append("test_acc")

    retained = [
        (name, column)
        for name, column in zip(names, columns, strict=True)
        if np.nanstd(column) > np.finfo(float).eps
    ]
    if not retained or len(steps) < 4:
        end = int(steps[-1])
        return PhaseResult([end], [Phase(int(steps[0]), end, "plateau")], [x[0] for x in retained])
    names = [item[0] for item in retained]
    signal = np.column_stack([item[1] for item in retained])
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)

    stride = max(1, int(np.ceil(len(signal) / 5000)))
    fitted = signal[::stride]
    used_penalty = penalty if penalty is not None else 3.0 * fitted.shape[1] * np.log(len(fitted))
    # KernelCPD charges its empirical RBF cost on both sides of a boundary. Dividing the
    # statistical penalty by two calibrates that implementation-specific two-sided cost.
    kernel_penalty = used_penalty if penalty is not None else used_penalty / 2.0
    indices = rpt.KernelCPD(kernel="rbf", min_size=2).fit(fitted).predict(pen=kernel_penalty)
    end_indices = [min(index * stride, len(steps)) for index in indices]
    end_indices[-1] = len(steps)
    breakpoints = [int(steps[index - 1]) for index in end_indices]

    phases: list[Phase] = []
    start = 0
    for end in end_indices:
        segment = signal[start:end]
        label = _label_segment(segment, names)
        phases.append(Phase(int(steps[start]), int(steps[end - 1]), label))
        start = end
    return PhaseResult(breakpoints, phases, names)


def _label_segment(segment: np.ndarray, names: list[str]) -> str:
    if len(segment) < 2:
        return "plateau"
    changes = {name: segment[-1, index] - segment[0, index] for index, name in enumerate(names)}
    ranges = {name: np.ptp(segment[:, index]) for index, name in enumerate(names)}
    loss_change = changes.get("train_loss", 0.0)
    test_change = changes.get("test_acc", 0.0)
    if ranges.get("grad_norm", 0.0) > 3.0 or ranges.get("train_loss", 0.0) > 4.0:
        return "instability"
    if loss_change < -0.5 and test_change < 0.25:
        return "memorization"
    if abs(loss_change) < 0.35 and test_change > 0.5:
        return "generalization"
    if abs(loss_change) < 0.35 and abs(test_change) < 0.25:
        return "plateau"
    return "fitting"
