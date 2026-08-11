"""Flag mislabeled points with influence functions and audit a custom gradient."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec import FlightRecorder, IndexedDataset, read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.influence import InfluenceConfig, self_influence
from flightrec.probes.gradsanity import GradReport, check_gradients, check_model_graph
from flightrec.report import build_report
from flightrec.utils import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/quick-influence-sanity")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mislabels", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


class TinyNet(nn.Module):
    """A small double-precision classifier whose exact Hessian is cheap to solve against."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))

    def forward(self, inputs: Tensor) -> Tensor:
        """Return two-class logits."""
        return self.layers(inputs)


class ScaledSquare(torch.autograd.Function):
    """``y = x**2`` with a correct hand-written backward."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:  # type: ignore[override]
        """Save the input and square it."""
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        """Return the exact derivative ``2x``."""
        (value,) = ctx.saved_tensors
        return 2 * value * grad_output


class BrokenSquare(torch.autograd.Function):
    """The same forward pass with a backward that drops the factor of two."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:  # type: ignore[override]
        """Save the input and square it."""
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        """Return a deliberately incorrect derivative."""
        (value,) = ctx.saved_tensors
        return value * grad_output


class PartiallyDetached(nn.Module):
    """A model whose second branch is silently cut out of the autograd graph."""

    def __init__(self) -> None:
        super().__init__()
        self.tracked = nn.Linear(2, 1)
        self.orphan = nn.Linear(2, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Add a connected branch to a detached one."""
        return self.tracked(inputs) + self.orphan(inputs).detach()


def make_data(samples: int, mislabels: int, seed: int) -> tuple[Tensor, Tensor, np.ndarray]:
    """Create standardized two-blob data with a known set of flipped labels."""
    features, labels = make_blobs(
        n_samples=samples, centers=2, n_features=2, cluster_std=1.6, random_state=seed
    )
    features = StandardScaler().fit_transform(features)
    mask = np.zeros(samples, dtype=bool)
    flipped = np.random.default_rng(seed).choice(samples, mislabels, replace=False)
    mask[flipped] = True
    noisy = labels.copy()
    noisy[mask] = 1 - noisy[mask]
    return (
        torch.tensor(features, dtype=torch.float64),
        torch.tensor(noisy, dtype=torch.long),
        mask,
    )


def audit_gradients() -> dict[str, GradReport]:
    """Run the gradient sanity layer against correct, broken, and detached code."""
    coordinates = {"x": torch.tensor([0.4, -1.1, 2.3])}
    return {
        "correct_custom_function": check_gradients(
            lambda values: ScaledSquare.apply(values["x"]).sum(), coordinates
        ),
        "broken_backward": check_gradients(
            lambda values: BrokenSquare.apply(values["x"]).sum(), coordinates
        ),
        "detached_branch": check_model_graph(
            broken_model := PartiallyDetached(), broken_model(torch.ones(4, 2)).sum()
        ),
    }


def render_gradient_table(reports: dict[str, GradReport], output: Path) -> None:
    """Write the gradient audit as a standalone table figure."""

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3e}"

    figure = go.Figure(
        go.Table(
            header={
                "values": [
                    "check",
                    "finite-difference error",
                    "complex-step error",
                    "unreachable parameters",
                    "verdict",
                ],
                "align": "left",
            },
            cells={
                "values": [
                    list(reports),
                    [number(report.max_abs_err_fd) for report in reports.values()],
                    [number(report.max_abs_err_cs) for report in reports.values()],
                    [", ".join(report.unreachable_params) or "none" for report in reports.values()],
                    ["passed" if report.passed else "FLAGGED" for report in reports.values()],
                ],
                "align": "left",
            },
        )
    )
    figure.update_layout(template="plotly_white", title="Gradient sanity audit")
    figure.write_html(output, include_plotlyjs="inline")


def detector_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Summarize one detector against the known mislabel mask."""
    count = int(labels.sum())
    ranked = np.argsort(scores)[::-1][:count]
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "precision_at_mislabel_count": float(labels[ranked].mean()),
    }


def main() -> None:
    """Train, score both mislabel detectors, audit gradients, and write the report."""
    args = parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "scalars.jsonl").exists():
        raise FileExistsError(f"{run_dir} already contains a run; choose a fresh --run-dir")
    features, labels, noise_mask = make_data(args.samples, args.mislabels, args.seed)
    np.save(run_dir / "noise_mask.npy", noise_mask)
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    dataset = TensorDataset(features, labels)
    loader = DataLoader(IndexedDataset(dataset), batch_size=len(dataset), shuffle=False)
    model = TinyNet().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    # Full-batch steps make every recorded step a complete epoch of per-example observations,
    # and they leave the model near a stationary point, which is what influence functions assume.
    with FlightRecorder(model, run_dir, num_samples=len(dataset)) as recorder:
        for _step in range(args.steps):
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
            recorder.epoch_end()
    torch.save(model.state_dict(), run_dir / "final_model.pt")

    run = read_run(run_dir)
    dynamics = suspicion_score(compute_event_stats(run))
    # One conjugate-gradient solve per candidate: the exact variant, affordable at this size.
    influence = self_influence(
        model,
        criterion,
        DataLoader(dataset, batch_size=32, shuffle=False),
        DataLoader(dataset, batch_size=len(dataset), shuffle=False),
        InfluenceConfig(
            damping=args.damping,
            cg_tol=1e-8,
            cg_maxiter=200,
            hessian_batches=1,
            last_layers_only=False,
        ),
        "cpu",
    )
    np.savez_compressed(run_dir / "influence_results.npz", scores=influence, mask=noise_mask)

    reports = audit_gradients()
    render_gradient_table(reports, run_dir / "gradient_sanity.html")
    build_report(
        run_dir,
        run_dir / "report.html",
        {"influence": influence, "influence_label": "exact self-influence"},
    )
    with torch.no_grad():
        accuracy = float((model(features).argmax(1) == labels).double().mean().item())
    metrics = {
        "samples": args.samples,
        "steps": args.steps,
        "mislabels": int(noise_mask.sum()),
        "damping": args.damping,
        "runtime_seconds": time.perf_counter() - started,
        "final_train_accuracy_against_noisy_labels": accuracy,
        "detectors": {
            "forgetting_dynamics": detector_scores(noise_mask, dynamics),
            "self_influence": detector_scores(noise_mask, influence),
        },
        "detector_agreement_spearman": float(spearmanr(dynamics, influence).statistic),
        "gradient_sanity": {
            name: {
                "max_abs_err_fd": report.max_abs_err_fd,
                "max_abs_err_cs": report.max_abs_err_cs,
                "unreachable_params": report.unreachable_params,
                "passed": report.passed,
            }
            for name, report in reports.items()
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
