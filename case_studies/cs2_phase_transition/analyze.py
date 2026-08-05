"""Analyze the modular-addition delayed generalization transition."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from flightrec import read_run
from flightrec.analysis.events import compute_event_stats
from flightrec.analysis.phases import detect_phases

try:
    from .model import ModularAdditionTransformer
    from .train import modular_data
except ImportError:
    from model import ModularAdditionTransformer
    from train import modular_data


def parse_args() -> argparse.Namespace:
    """Parse analysis options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/cs2")
    return parser.parse_args()


def main() -> None:
    """Locate the transition and write timeline and per-example figures."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    run = read_run(run_dir)
    result = detect_phases(run)
    kinds = run.scalars["kind"]
    valid = (kinds == "eval") & np.isfinite(run.scalars["test_acc"].astype(float))
    eval_steps = run.scalars["step"][valid].astype(int)
    test_acc = run.scalars["test_acc"][valid].astype(float)
    train_acc = run.scalars["train_acc"][valid].astype(float)
    low = eval_steps[np.flatnonzero(test_acc >= 0.05)[0]]
    high = eval_steps[np.flatnonzero(test_acc >= 0.95)[0]]
    train_learned = eval_steps[np.flatnonzero(train_acc >= 0.99)[0]]
    test_generalized = eval_steps[np.flatnonzero(test_acc >= 0.90)[0]]
    separation = int(test_generalized - train_learned)
    found = any(low <= point <= high for point in result.breakpoints[:-1])
    print(f"Transition window: {low}..{high}; phase breakpoint detected: {found}")
    print(f"Train/test delayed generalization separation: {separation} steps")
    measured = {
        "transition_start_step": int(low),
        "transition_end_step": int(high),
        "train_99_step": int(train_learned),
        "test_90_step": int(test_generalized),
        "separation_steps": separation,
        "breakpoint_in_transition": found,
        "breakpoints": result.breakpoints,
    }
    (run_dir / "analysis_metrics.json").write_text(json.dumps(measured, indent=2))
    if not found:
        raise AssertionError(
            "no automatically detected breakpoint lies in the delayed generalization transition"
        )
    if separation < 3000:
        raise AssertionError(
            "run did not exhibit the required 3,000-step delayed generalization separation"
        )

    figure = make_subplots(rows=3, cols=1, shared_xaxes=True)
    if "train_acc" in run.scalars:
        figure.add_trace(go.Scatter(x=eval_steps, y=train_acc, name="train accuracy"), row=1, col=1)
    figure.add_trace(go.Scatter(x=eval_steps, y=test_acc, name="test accuracy"), row=1, col=1)
    step_mask = kinds == "step"
    figure.add_trace(
        go.Scatter(
            x=run.scalars["step"][step_mask],
            y=run.scalars["param_norm"][step_mask],
            name="weight norm",
        ),
        row=2,
        col=1,
    )
    if run.spectrum_steps is not None and run.eigs_high is not None:
        figure.add_trace(
            go.Scatter(
                x=run.spectrum_steps, y=np.max(run.eigs_high, axis=1), name="top eigenvalue"
            ),
            row=3,
            col=1,
        )
    for phase in result.phases:
        figure.add_vrect(
            x0=max(1, phase.start_step),
            x1=phase.end_step,
            fillcolor="rgba(80,120,200,.08)",
            line_width=0,
            annotation_text=phase.label,
            row="all",
            col=1,
        )
    figure.update_xaxes(type="log")
    figure.update_layout(template="plotly_white", title="delayed generalization timeline")
    figure.write_html(run_dir / "phase_transition_timeline.html", include_plotlyjs="inline")

    if run.correct is None:
        return
    stats = compute_event_stats(run)
    order = np.argsort(stats.first_learned)
    heatmap = go.Figure(go.Heatmap(z=run.correct[:, order].T, colorscale="Blues"))
    heatmap.update_layout(
        template="plotly_white", xaxis_title="recording epoch", yaxis_title="sample"
    )
    heatmap.write_html(run_dir / "learning_heatmap.html", include_plotlyjs="inline")

    config = json.loads((run_dir / "training_config.json").read_text())
    _, test_set = modular_data(config["p"], config["train_frac"], config["seed"])
    test_loader = DataLoader(test_set, batch_size=1024)
    margin_figure = go.Figure()
    for name in ("pre", "mid", "post"):
        checkpoint = run_dir / f"checkpoint_{name}.pt"
        if not checkpoint.exists():
            continue
        model = ModularAdditionTransformer(config["p"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        margins = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                logits = model(inputs)
                true = logits.gather(1, targets[:, None]).squeeze(1)
                logits.scatter_(1, targets[:, None], -torch.inf)
                margins.extend((true - logits.max(1).values).numpy())
        margin_figure.add_trace(go.Histogram(x=margins, name=name, opacity=0.5))
    margin_figure.update_layout(template="plotly_white", barmode="overlay", title="Test margins")
    margin_figure.write_html(run_dir / "test_margins.html", include_plotlyjs="inline")


if __name__ == "__main__":
    main()
