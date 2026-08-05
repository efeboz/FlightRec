"""Plotly figure factories used by the HTML report."""

from types import MappingProxyType

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flightrec.analysis.events import EventStats
from flightrec.analysis.phases import PhaseResult
from flightrec.storage import RunData

TEMPLATE = "plotly_white"
PHASE_COLORS = MappingProxyType(
    {
        "memorization": "rgba(255, 193, 7, 0.12)",
        "generalization": "rgba(40, 167, 69, 0.12)",
        "plateau": "rgba(108, 117, 125, 0.10)",
        "instability": "rgba(220, 53, 69, 0.12)",
        "fitting": "rgba(0, 123, 255, 0.10)",
    }
)


def _phase_bands(figure: go.Figure, phases: PhaseResult | None) -> None:
    if phases is None:
        return
    for phase in phases.phases:
        figure.add_vrect(
            x0=phase.start_step,
            x1=phase.end_step,
            fillcolor=PHASE_COLORS.get(phase.label, "rgba(0,0,0,0.05)"),
            line_width=0,
            annotation_text=phase.label,
            annotation_position="top left",
        )


def timeline_figure(run: RunData, phases: PhaseResult | None = None) -> go.Figure:
    """Build a training scalar timeline."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    steps = run.scalars.get("step", np.empty(0))
    kinds = run.scalars.get("kind", np.full(len(steps), "step"))
    for field, name, secondary in (
        ("loss", "train loss", False),
        ("test_loss", "test loss", False),
        ("test_acc", "test accuracy", True),
        ("grad_norm", "gradient norm", True),
        ("lr", "learning rate", True),
    ):
        if field not in run.scalars:
            continue
        values = run.scalars[field].astype(float)
        mask = np.isfinite(values)
        if field == "loss":
            mask &= kinds == "step"
        figure.add_trace(
            go.Scatter(x=steps[mask], y=values[mask], name=name), secondary_y=secondary
        )
    _phase_bands(figure, phases)
    figure.update_layout(template=TEMPLATE, title="Training timeline", xaxis_title="step")
    return figure


def spectrum_figure(run: RunData, phases: PhaseResult | None = None) -> go.Figure:
    """Build extremal Hessian eigenvalue trajectories."""
    figure = go.Figure()
    if run.spectrum_steps is not None and run.eigs_high is not None:
        figure.add_trace(
            go.Scatter(x=run.spectrum_steps, y=np.max(run.eigs_high, axis=1), name="largest")
        )
    if run.spectrum_steps is not None and run.eigs_low is not None:
        figure.add_trace(
            go.Scatter(x=run.spectrum_steps, y=np.min(run.eigs_low, axis=1), name="smallest")
        )
    _phase_bands(figure, phases)
    figure.update_layout(template=TEMPLATE, title="Hessian spectrum", xaxis_title="step")
    return figure


def forgetting_figure(stats: EventStats) -> go.Figure:
    """Build a histogram of forgetting counts."""
    figure = go.Figure(go.Histogram(x=stats.forgetting_count))
    figure.update_layout(template=TEMPLATE, title="Forgetting events", xaxis_title="count")
    return figure


def first_learned_figure(stats: EventStats) -> go.Figure:
    """Build the first-learned epoch distribution."""
    learned = stats.first_learned[stats.first_learned >= 0]
    figure = go.Figure(go.Histogram(x=learned))
    figure.update_layout(template=TEMPLATE, title="First learned", xaxis_title="epoch")
    return figure


def margin_forgetting_figure(stats: EventStats) -> go.Figure:
    """Build a 2-D density plot of margin against forgetting."""
    valid = np.isfinite(stats.mean_margin)
    figure = go.Figure(
        go.Histogram2d(
            x=stats.mean_margin[valid],
            y=stats.forgetting_count[valid],
            colorscale="Blues",
        )
    )
    figure.update_layout(
        template=TEMPLATE,
        title="Margin vs. forgetting",
        xaxis_title="mean margin",
        yaxis_title="forgetting count",
    )
    return figure


def influence_figure(
    suspicion: np.ndarray,
    influence: np.ndarray,
    rho: float,
    influence_label: str = "self-influence",
) -> go.Figure:
    """Build an influence-versus-dynamics scatter plot."""
    figure = go.Figure(go.Scatter(x=suspicion, y=influence, mode="markers"))
    figure.update_layout(
        template=TEMPLATE,
        title=f"Influence vs. suspicion (Spearman rho={rho:.3f})",
        xaxis_title="suspicion score",
        yaxis_title=influence_label,
    )
    return figure
