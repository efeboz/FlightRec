"""Vectorized per-example learning and forgetting statistics."""

from dataclasses import dataclass

import numpy as np

from flightrec.storage import RunData


@dataclass
class EventStats:
    """Per-sample learning-event summary."""

    first_learned: np.ndarray
    forgetting_count: np.ndarray
    never_learned: np.ndarray
    final_correct: np.ndarray
    mean_margin: np.ndarray


def compute_event_stats(run: RunData) -> EventStats:
    """Compute learning events while carrying state across unseen epochs."""
    if run.correct is None:
        raise ValueError("run contains no per-example correctness data")
    correct = np.asarray(run.correct, dtype=np.uint8)
    if correct.ndim != 2:
        raise ValueError("correct must have shape [epochs, samples]")
    epochs, samples = correct.shape
    if epochs == 0:
        empty_i = np.full(samples, -1, dtype=np.int32)
        return EventStats(
            empty_i,
            np.zeros(samples, np.int32),
            np.ones(samples, bool),
            np.zeros(samples, bool),
            np.full(samples, np.nan, np.float32),
        )

    observed = correct != 255
    epoch_index = np.broadcast_to(np.arange(epochs)[:, None], correct.shape)
    last_observed = np.maximum.accumulate(np.where(observed, epoch_index, -1), axis=0)
    safe_index = np.maximum(last_observed, 0)
    carried = np.take_along_axis(correct, safe_index, axis=0) == 1
    carried[last_observed < 0] = False

    previous = np.vstack([np.zeros((1, samples), dtype=bool), carried[:-1]])
    learned = observed & carried & ~previous
    forgotten = observed & ~carried & previous
    ever_learned = learned.any(axis=0)
    first = np.where(ever_learned, learned.argmax(axis=0), -1).astype(np.int32)
    counts = forgotten.sum(axis=0, dtype=np.int32)

    if run.margin is None:
        mean_margin = np.full(samples, np.nan, dtype=np.float32)
    else:
        margins = np.asarray(run.margin, dtype=np.float32)
        valid = ~np.isnan(margins)
        totals = np.nansum(margins, axis=0)
        numbers = valid.sum(axis=0)
        mean_margin = np.divide(
            totals,
            numbers,
            out=np.full(samples, np.nan, dtype=np.float32),
            where=numbers > 0,
        ).astype(np.float32)
    return EventStats(first, counts, ~ever_learned, carried[-1], mean_margin)


def _rank(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    average_ranks = starts + (counts - 1) / 2.0
    return average_ranks[inverse]


def suspicion_score(stats: EventStats) -> np.ndarray:
    """Rank-combine forgetting, late learning, and low-margin evidence."""
    count = len(stats.first_learned)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    late = stats.first_learned.astype(np.float64)
    late[stats.never_learned] = np.inf
    margin = stats.mean_margin.astype(np.float64)
    margin[np.isnan(margin)] = -np.inf
    combined = (_rank(stats.forgetting_count) + _rank(late) + _rank(-margin)) / 3.0
    span = float(combined.max() - combined.min())
    if span == 0.0:
        return np.zeros(count, dtype=np.float64)
    return (combined - combined.min()) / span
