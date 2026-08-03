"""Lanczos spectrum and stochastic trace estimates."""

import numpy as np
from scipy.sparse.linalg import eigsh

from flightrec.probes.hvp import HessianOperator


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
