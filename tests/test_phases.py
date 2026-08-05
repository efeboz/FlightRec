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


def test_detects_accuracy_transition_with_long_plateau_and_downsampling():
    rng = np.random.default_rng(5)
    length = 6000
    steps = np.arange(length, dtype=float)
    loss = 0.03 + 4.0 * np.exp(-steps / 250) + rng.normal(0, 0.003, length)
    grad = 0.05 + 1.5 * np.exp(-steps / 350) + rng.normal(0, 0.005, length)
    param = 25.0 + 75.0 * np.exp(-steps / 1600) + rng.normal(0, 0.02, length)
    test_acc = 1.0 / (1.0 + np.exp(-(steps - 1500) / 100))
    run = RunData(
        {},
        {
            "kind": np.full(length, "step"),
            "step": steps,
            "loss": loss,
            "grad_norm": grad,
            "param_norm": param,
            "test_acc": test_acc,
        },
    )
    result = detect_phases(run)
    transition_start = int(np.flatnonzero(test_acc >= 0.05)[0])
    transition_end = int(np.flatnonzero(test_acc >= 0.95)[0])
    assert any(transition_start <= point <= transition_end for point in result.breakpoints[:-1])
