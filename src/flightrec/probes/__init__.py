"""Numerical probes for PyTorch models."""

from flightrec.probes.curvature import hutchinson_trace, lanczos_ritz_spectrum, lanczos_spectrum
from flightrec.probes.gradsanity import GradReport, check_gradients, check_model_graph
from flightrec.probes.hvp import HessianOperator, hvp

__all__ = [
    "GradReport",
    "HessianOperator",
    "check_gradients",
    "check_model_graph",
    "hutchinson_trace",
    "hvp",
    "lanczos_ritz_spectrum",
    "lanczos_spectrum",
]
