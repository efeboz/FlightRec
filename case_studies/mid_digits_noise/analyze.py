"""Score both mislabel detectors on the digit run and build the illustrated report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec import read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.influence import InfluenceConfig, influence_on
from flightrec.analysis.phases import detect_phases
from flightrec.report import build_report
from flightrec.utils import seed_everything

try:
    from .train import DigitCNN, load_digit_data
except ImportError:
    from train import DigitCNN, load_digit_data


def parse_args() -> argparse.Namespace:
    """Parse analysis options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/mid-digits")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--cg-maxiter", type=int, default=200)
    parser.add_argument("--hessian-batches", type=int, default=4)
    parser.add_argument("--reference-size", type=int, default=256)
    parser.add_argument("--skip-influence", action="store_true")
    return parser.parse_args()


def rank01(values: np.ndarray) -> np.ndarray:
    """Map values to average ranks in [0, 1], preserving ties."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = (starts + (counts - 1) / 2.0)[inverse]
    return ranks / max(1, len(values) - 1)


def thumbnail_source(images: np.ndarray):
    """Return a callable turning a sample index into a legible 64x64 grayscale thumbnail."""

    def render(index: int) -> np.ndarray:
        pixels = np.clip(images[index, 0] * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(np.repeat(pixels, 8, axis=0), 8, axis=1)

    return render


def main() -> None:
    """Evaluate detectors, write metrics and PR curves, and render the report."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "training_config.json").read_text())
    seed_everything(config["seed"])
    run = read_run(run_dir)
    noise_mask = np.load(run_dir / "noise_mask.npy")
    images = np.load(run_dir / "train_images.npy")
    noisy_labels = np.load(run_dir / "noisy_labels.npy")
    original_labels = np.load(run_dir / "original_labels.npy")

    stats = compute_event_stats(run)
    dynamics = suspicion_score(stats)
    detectors = {"forgetting": dynamics}
    influence: np.ndarray | None = None
    influence_seconds = 0.0

    if not args.skip_influence:
        started = time.perf_counter()
        _, _, test_x, test_y = load_digit_data(config["seed"])
        train_x = torch.tensor(images)
        train_y = torch.tensor(noisy_labels, dtype=torch.long)
        dataset = TensorDataset(train_x, train_y)
        model = DigitCNN()
        model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu"))
        # A clean validation batch turns influence into "which training points raise held-out
        # loss", which is the same scalable formulation the CIFAR-10 case study uses.
        reference = (test_x[: args.reference_size], test_y[: args.reference_size])
        generator = torch.Generator().manual_seed(config["seed"])
        influence = influence_on(
            model,
            nn.CrossEntropyLoss(),
            reference,
            DataLoader(dataset, batch_size=128, shuffle=False),
            DataLoader(dataset, batch_size=128, shuffle=True, generator=generator),
            InfluenceConfig(
                damping=args.damping,
                cg_maxiter=args.cg_maxiter,
                hessian_batches=args.hessian_batches,
                last_layers_only=True,
            ),
            "cpu",
        )
        influence_seconds = time.perf_counter() - started
        np.savez_compressed(run_dir / "influence_results.npz", scores=influence)
        detectors["influence"] = rank01(influence)
        detectors["combined"] = (rank01(dynamics) + rank01(influence)) / 2.0

    count = int(noise_mask.sum())
    measured: dict[str, object] = {}
    figure = go.Figure()
    print("| detector | AP | precision@k |")
    print("|---|---:|---:|")
    for name, score in detectors.items():
        average_precision = float(average_precision_score(noise_mask, score))
        top = np.argsort(score)[::-1][:count]
        precision_at_k = float(noise_mask[top].mean())
        measured[name] = {
            "average_precision": average_precision,
            "precision_at_k": precision_at_k,
        }
        print(f"| {name} | {average_precision:.4f} | {precision_at_k:.4f} |")
        precision, recall, _ = precision_recall_curve(noise_mask, score)
        figure.add_trace(go.Scatter(x=recall, y=precision, name=name))
    figure.add_hline(y=float(noise_mask.mean()), line_dash="dot", annotation_text="chance")
    figure.update_layout(
        template="plotly_white",
        title="Mislabel detection on corrupted digits",
        xaxis_title="recall",
        yaxis_title="precision",
    )
    figure.write_html(run_dir / "pr_curves.html", include_plotlyjs="inline")

    phases = detect_phases(run)
    summary = {
        "detectors": measured,
        "injected_mislabels": count,
        "training_samples": int(len(noise_mask)),
        "influence_seconds": influence_seconds,
        "phases": [
            {"start_step": phase.start_step, "end_step": phase.end_step, "label": phase.label}
            for phase in phases.phases
        ],
        "phase_signals": phases.signals_used,
        "spectrum_probes": 0 if run.eigs_high is None else int(len(run.eigs_high)),
    }
    if run.eigs_high is not None and run.eigs_low is not None:
        summary["eigenvalue_range"] = [float(np.min(run.eigs_low)), float(np.max(run.eigs_high))]
    (run_dir / "analysis_metrics.json").write_text(json.dumps(summary, indent=2))

    def describe(index: int) -> str:
        if noise_mask[index]:
            return f"label {noisy_labels[index]}, truly {original_labels[index]}"
        return f"label {noisy_labels[index]}, clean"

    extras: dict[str, object] = {
        "images": thumbnail_source(images),
        "image_labels": describe,
    }
    if influence is not None:
        extras["influence"] = influence
        extras["influence_label"] = "validation influence"
    build_report(run_dir, run_dir / "report.html", extras)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
