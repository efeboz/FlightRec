"""Run a compact noisy-label experiment and render its diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.datasets import make_moons
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec import FlightRecorder, IndexedDataset, read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.report import build_report
from flightrec.utils import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/quick-label-noise")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noise-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_data(
    samples: int, noise_rate: float, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray]:
    """Create standardized two-moons data with train-only label corruption."""
    features, labels = make_moons(n_samples=samples, noise=0.18, random_state=seed)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(train_y), dtype=bool)
    changed = rng.choice(len(mask), int(round(noise_rate * len(mask))), replace=False)
    mask[changed] = True
    noisy_y = train_y.copy()
    noisy_y[mask] = 1 - noisy_y[mask]
    return (
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(noisy_y, dtype=torch.long),
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.long),
        mask,
    )


class TinyClassifier(nn.Module):
    """A small MLP with enough capacity to eventually fit corrupted labels."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        """Return two-class logits."""
        return self.layers(inputs)


@torch.no_grad()
def accuracy(model: nn.Module, features: Tensor, labels: Tensor) -> float:
    """Return classification accuracy."""
    return float((model(features).argmax(1) == labels).float().mean().item())


def render_map(
    model: nn.Module,
    features: Tensor,
    noisy_labels: Tensor,
    noise_mask: np.ndarray,
    scores: np.ndarray,
    output: Path,
) -> None:
    """Render the decision surface, injected noise, and highest suspicion scores."""
    x_values = np.linspace(
        float(features[:, 0].min()) - 0.5, float(features[:, 0].max()) + 0.5, 140
    )
    y_values = np.linspace(
        float(features[:, 1].min()) - 0.5, float(features[:, 1].max()) + 0.5, 140
    )
    xx, yy = np.meshgrid(x_values, y_values)
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        probability = model(grid).softmax(1)[:, 1].reshape(xx.shape).numpy()
    figure = go.Figure(
        go.Contour(
            x=x_values,
            y=y_values,
            z=probability,
            colorscale="RdBu",
            opacity=0.45,
            contours={"start": 0.0, "end": 1.0, "size": 0.1},
            colorbar={"title": "P(class 1)"},
        )
    )
    clean = ~noise_mask
    figure.add_trace(
        go.Scatter(
            x=features[clean, 0],
            y=features[clean, 1],
            mode="markers",
            name="clean training label",
            marker={"color": noisy_labels[clean], "colorscale": "Viridis", "size": 7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=features[noise_mask, 0],
            y=features[noise_mask, 1],
            mode="markers",
            name="injected label noise",
            marker={"color": "black", "symbol": "x", "size": 9},
        )
    )
    top = np.argsort(scores)[::-1][: int(noise_mask.sum())].copy()
    figure.add_trace(
        go.Scatter(
            x=features[top, 0],
            y=features[top, 1],
            mode="markers",
            name="top FlightRec flags",
            marker={"color": "rgba(0,0,0,0)", "line": {"color": "gold", "width": 2}, "size": 14},
        )
    )
    figure.update_layout(
        template="plotly_white",
        title="Quick noisy-label recovery: decision map and flagged points",
        xaxis_title="standardized feature 1",
        yaxis_title="standardized feature 2",
    )
    figure.write_html(output, include_plotlyjs="inline")


def main() -> None:
    """Train the compact example and write metrics plus two visualizations."""
    args = parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "scalars.jsonl").exists():
        raise FileExistsError(f"{run_dir} already contains a run; choose a fresh --run-dir")
    train_x, train_y, test_x, test_y, noise_mask = make_data(
        args.samples, args.noise_rate, args.seed
    )
    np.save(run_dir / "noise_mask.npy", noise_mask)
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))
    loader = DataLoader(
        IndexedDataset(TensorDataset(train_x, train_y)), batch_size=64, shuffle=True
    )
    model = TinyClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    with FlightRecorder(model, run_dir, num_samples=len(train_x)) as recorder:
        for _epoch in range(args.epochs):
            for inputs, targets, indices in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                recorder.record_step(
                    loss=loss,
                    logits=logits,
                    targets=targets,
                    sample_indices=indices,
                    lr=optimizer.param_groups[0]["lr"],
                )
                optimizer.step()
            recorder.record_eval(test_acc=accuracy(model, test_x, test_y))
            recorder.epoch_end()
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    run = read_run(run_dir)
    scores = suspicion_score(compute_event_stats(run))
    k = int(noise_mask.sum())
    average_precision = float(average_precision_score(noise_mask, scores))
    precision_at_k = float(noise_mask[np.argsort(scores)[::-1][:k]].mean())
    runtime = time.perf_counter() - started
    metrics = {
        "samples": args.samples,
        "training_samples": len(train_x),
        "epochs": args.epochs,
        "runtime_seconds": runtime,
        "test_accuracy": accuracy(model, test_x, test_y),
        "average_precision": average_precision,
        "precision_at_noise_count": precision_at_k,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    render_map(model, train_x, train_y, noise_mask, scores, run_dir / "label_noise_map.html")
    build_report(run_dir, run_dir / "report.html")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
