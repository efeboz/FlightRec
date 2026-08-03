import numpy as np

from flightrec.analysis.phases import detect_phases
from flightrec.storage import RunData


def test_detects_synthetic_change_points():
    rng = np.random.default_rng(2)
    length = 400
    steps = np.arange(length, dtype=float)
    loss = np.r_[
        np.linspace(5, 2, 100),
        np.linspace(2, 1.8, 100),
        np.linspace(1.8, 0.1, 100),
        np.full(100, 0.1),
    ] + rng.normal(0, 0.015, length)
    grad = np.r_[np.ones(100), np.full(100, 3), np.full(100, 0.6), np.full(100, 0.1)]
    grad += rng.normal(0, 0.02, length)
    param = np.r_[
        np.linspace(1, 2, 100),
        np.linspace(2, 4, 100),
        np.linspace(4, 3, 100),
        np.full(100, 3),
    ] + rng.normal(0, 0.01, length)
    run = RunData(
        {},
        {
            "kind": np.full(length, "step"),
            "step": steps,
            "loss": loss,
            "grad_norm": grad,
            "param_norm": param,
        },
    )
    result = detect_phases(run)
    detected = result.breakpoints[:-1]
    for truth in (99, 199, 299):
        assert any(abs(point - truth) <= 0.03 * length for point in detected)
