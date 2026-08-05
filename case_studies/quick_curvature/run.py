"""Train a tiny classifier while recording repeated Hessian spectra."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from flightrec import FlightRecorder, read_run
from flightrec.report import build_report
from flightrec.utils import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/quick-curvature")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--spectrum-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


class CurvatureMLP(nn.Module):
    """Tiny smooth classifier whose complete Hessian is inexpensive to probe."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        """Return two-class logits."""
        return self.layers(inputs)


def make_data(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a deterministic train/test split of a nonlinear toy problem."""
    features, labels = make_moons(n_samples=320, noise=0.2, random_state=seed)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=seed
    )
    scaler = StandardScaler().fit(train_x)
    return (
        torch.tensor(scaler.transform(train_x), dtype=torch.float64),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(scaler.transform(test_x), dtype=torch.float64),
        torch.tensor(test_y, dtype=torch.long),
    )


def render_curvature(run_dir: Path) -> None:
    """Render extremal eigenvalues and loss on a shared step axis."""
    run = read_run(run_dir)
    assert run.spectrum_steps is not None
    assert run.eigs_low is not None and run.eigs_high is not None
    step_mask = run.scalars["kind"] == "step"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=run.scalars["step"][step_mask],
            y=run.scalars["loss"][step_mask],
            name="training loss",
            yaxis="y",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=run.spectrum_steps,
            y=np.max(run.eigs_high, axis=1),
            name="largest Hessian eigenvalue",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=run.spectrum_steps,
            y=np.min(run.eigs_low, axis=1),
            name="smallest Hessian eigenvalue",
            yaxis="y2",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title="Quick curvature trajectory",
        xaxis={"title": "optimization step"},
        yaxis={"title": "cross-entropy"},
        yaxis2={"title": "Hessian eigenvalue", "overlaying": "y", "side": "right"},
    )
    figure.write_html(run_dir / "curvature_timeline.html", include_plotlyjs="inline")


def main() -> None:
    """Train the tiny model and generate a complete report."""
    args = parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "scalars.jsonl").exists():
        raise FileExistsError(f"{run_dir} already contains a run; choose a fresh --run-dir")
    train_x, train_y, test_x, test_y = make_data(args.seed)
    model = CurvatureMLP().double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.8)

    def probe_loss(probe_model: nn.Module, batch: object) -> Tensor:
        inputs, targets = batch
        return criterion(probe_model(inputs), targets)

    with FlightRecorder(
        model,
        run_dir,
        num_samples=len(train_x),
        spectrum_every=args.spectrum_every,
        spectrum_k=2,
        probe_device="cpu",
        probe_loss_fn=probe_loss,
        probe_batch=(train_x, train_y),
    ) as recorder:
        indices = torch.arange(len(train_x))
        for _step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            recorder.record_step(
                loss=loss,
                logits=logits,
                targets=train_y,
                sample_indices=indices,
                lr=optimizer.param_groups[0]["lr"],
            )
            optimizer.step()
            with torch.no_grad():
                test_acc = float((model(test_x).argmax(1) == test_y).double().mean().item())
            recorder.record_eval(test_acc=test_acc)
            recorder.epoch_end()
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    render_curvature(run_dir)
    build_report(run_dir, run_dir / "report.html")
    run = read_run(run_dir)
    assert run.eigs_low is not None and run.eigs_high is not None
    metrics = {
        "steps": args.steps,
        "spectrum_probes": len(run.eigs_high),
        "runtime_seconds": time.perf_counter() - started,
        "final_test_accuracy": float(run.scalars["test_acc"][-1]),
        "minimum_eigenvalue": float(np.min(run.eigs_low)),
        "maximum_eigenvalue": float(np.max(run.eigs_high)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
