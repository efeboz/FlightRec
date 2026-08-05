import time

import numpy as np

from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.storage import RunData


def test_events_carry_unseen_state():
    correct = np.array([[0, 1, 255], [1, 255, 0], [255, 0, 1], [0, 1, 1]], dtype=np.uint8)
    margin = np.where(correct == 255, np.nan, correct * 2 - 1).astype(np.float16)
    stats = compute_event_stats(RunData({}, {}, correct, margin))
    np.testing.assert_array_equal(stats.first_learned, [1, 0, 2])
    np.testing.assert_array_equal(stats.forgetting_count, [1, 1, 0])
    np.testing.assert_array_equal(stats.final_correct, [False, True, True])
    scores = suspicion_score(stats)
    assert scores.shape == (3,)
    assert np.all((0 <= scores) & (scores <= 1))


def test_events_vectorized_large():
    rng = np.random.default_rng(0)
    correct = rng.choice(np.array([0, 1, 255], np.uint8), size=(100, 50_000), p=[0.45, 0.45, 0.1])
    margin = rng.normal(size=correct.shape).astype(np.float16)
    margin[correct == 255] = np.nan
    start = time.perf_counter()
    stats = compute_event_stats(RunData({}, {}, correct, margin))
    assert time.perf_counter() - start < 1.0
    assert stats.first_learned.shape == (50_000,)


def test_never_learned_ranks_as_maximal_forgetting_evidence():
    correct = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    margin = np.array(
        [
            [-1.0, 1.0, 2.0],
            [-1.0, -1.0, 2.0],
            [-1.0, 1.0, 2.0],
            [-1.0, -1.0, 2.0],
        ],
        dtype=np.float16,
    )
    stats = compute_event_stats(RunData({}, {}, correct, margin))
    assert stats.forgetting_count.tolist() == [0, 2, 0]
    scores = suspicion_score(stats)
    assert scores[0] > scores[1] > scores[2]
