"""Post-training analysis routines."""

from flightrec.analysis.events import EventStats, compute_event_stats, suspicion_score
from flightrec.analysis.phases import Phase, PhaseResult, detect_phases

__all__ = [
    "EventStats",
    "Phase",
    "PhaseResult",
    "compute_event_stats",
    "detect_phases",
    "suspicion_score",
]
