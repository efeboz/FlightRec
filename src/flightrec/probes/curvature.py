"""Lanczos spectrum and stochastic trace estimates."""

import numpy as np
from scipy.sparse.linalg import eigsh

from flightrec.probes.hvp import HessianOperator


def _vector_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a checked dot product without platform BLAS status-flag defects."""
    value = 0.0
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        for start in range(0, len(left), 1_000_000):
            stop = min(start + 1_000_000, len(left))
            value += float(np.sum(np.multiply(left[start:stop], right[start:stop])))
    if not np.isfinite(value):
        raise FloatingPointError("Lanczos vector reduction is non-finite")
    return value


def _vector_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, _vector_dot(vector, vector))))


def lanczos_spectrum(op: HessianOperator, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` smallest and largest algebraic Hessian eigenvalues."""
    if k < 1:
        raise ValueError("k must be positive")
    size = op.shape[0]
    take = min(k, size)
    if size <= 2 * k:
        eye = np.eye(size)
        dense = np.column_stack([op.matvec(eye[:, index]) for index in range(size)])
        values = np.linalg.eigvalsh((dense + dense.T) / 2.0)
    else:
        values = eigsh(op, k=2 * k, which="BE", return_eigenvectors=False)
        values.sort()
    return values[:take].copy(), values[-take:].copy()


def lanczos_ritz_spectrum(
    op: HessianOperator,
    k: int = 1,
    steps: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate both algebraic Hessian ends with a fixed Lanczos budget.

    Unlike :func:`lanczos_spectrum`, this telemetry-oriented variant has a predictable number of
    matrix-vector products. Full reorthogonalization keeps the small projected tridiagonal stable.
    The returned values are Ritz estimates, not fully converged ARPACK eigenvalues.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if steps < 2 * k:
        raise ValueError("steps must be at least 2 * k")
    size = op.shape[0]
    iterations = min(size, steps)
    generator = np.random.default_rng(seed)
    current = generator.standard_normal(size)
    current /= _vector_norm(current)
    basis: list[np.ndarray] = []
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    previous = np.zeros_like(current)
    previous_beta = 0.0
    for _ in range(iterations):
        basis.append(current.copy())
        product = np.asarray(op.matvec(current), dtype=np.float64)
        residual = product - previous_beta * previous
        alpha = _vector_dot(current, residual)
        residual -= alpha * current
        # A second pass is inexpensive at the small telemetry budgets used here and prevents
        # duplicate Ritz values after loss of Lanczos-vector orthogonality.
        for vector in basis:
            residual -= _vector_dot(vector, residual) * vector
        beta = _vector_norm(residual)
        diagonal.append(alpha)
        if len(diagonal) < iterations:
            off_diagonal.append(beta)
        if beta <= np.finfo(float).eps * max(1.0, _vector_norm(product)):
            off_diagonal = off_diagonal[: len(diagonal) - 1]
            break
        previous, current = current, residual / beta
        previous_beta = beta
    projected = np.diag(diagonal)
    if len(diagonal) > 1:
        off = np.asarray(off_diagonal[: len(diagonal) - 1])
        projected += np.diag(off, 1) + np.diag(off, -1)
    values = np.linalg.eigvalsh(projected)
    take = min(k, len(values))
    return values[:take].copy(), values[-take:].copy()


def hutchinson_trace(op: HessianOperator, n_probes: int = 32, seed: int = 0) -> float:
    """Estimate ``tr(H)`` with seeded Rademacher probes."""
    if n_probes < 1:
        raise ValueError("n_probes must be positive")
    generator = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_probes):
        vector = generator.choice(np.array([-1.0, 1.0]), size=op.shape[0])
        total += float(vector @ op.matvec(vector))
    return total / n_probes
