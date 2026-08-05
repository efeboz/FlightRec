"""Post-training analysis routines."""

from flightrec.analysis.events import EventStats, compute_event_stats, suspicion_score
from flightrec.analysis.influence import InfluenceConfig, influence_on, self_influence
from flightrec.analysis.phases import Phase, PhaseResult, detect_phases

__all__ = [
    "EventStats",
    "InfluenceConfig",
    "Phase",
    "PhaseResult",
    "compute_event_stats",
    "detect_phases",
    "influence_on",
    "self_influence",
    "suspicion_score",
]
