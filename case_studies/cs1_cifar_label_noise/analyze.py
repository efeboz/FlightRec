"""Analyze noisy-label recovery and produce the CS1 report."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from model import resnet18_cifar
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from flightrec import read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.influence import InfluenceConfig, self_influence
from flightrec.report import build_report
from flightrec.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse analysis options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/cs1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--influence-candidates", type=int, default=5000)
    parser.add_argument("--skip-influence", action="store_true")
    return parser.parse_args()


def rank01(values: np.ndarray) -> np.ndarray:
    """Map values to stable ranks in [0, 1]."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(len(values))
    return ranks / max(1, len(values) - 1)


def main() -> None:
    """Evaluate detectors and emit interactive reports."""
    args = parse_args()
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    run = read_run(run_dir)
    noise = np.load(run_dir / "noise_mask.npy")
    stats = compute_event_stats(run)
    dynamics = suspicion_score(stats)
    influence_full = np.full(len(noise), np.nan)

    if not args.skip_influence:
        config = json.loads((run_dir / "training_config.json").read_text())
        subset_indices = np.load(run_dir / "subset_indices.npy")
        original = np.load(run_dir / "original_labels.npy")
        noisy_labels = original.copy()
        # Recover injected labels from saved run metadata when possible; noise identity is enough
        # for detector evaluation, and deterministic regeneration preserves training labels.
        rng = np.random.default_rng(config["seed"])
        all_indices = rng.permutation(50000)
        _ = all_indices[: config.get("subset") or 50000]
        chosen = rng.choice(
            len(subset_indices),
            int(round(config["noise_rate"] * len(subset_indices))),
            replace=False,
        )
        noisy_labels[chosen] = (noisy_labels[chosen] + rng.integers(1, 10, len(chosen))) % 10
        base = CIFAR10(args.data_dir, train=True, download=True, transform=ToTensor())
        tensors = torch.stack([base[int(index)][0] for index in subset_indices])
        dataset = TensorDataset(tensors, torch.as_tensor(noisy_labels))
        top = np.argsort(dynamics)[::-1][: min(2000, len(noise))]
        remaining = np.setdiff1d(np.arange(len(noise)), top)
        count = min(max(0, args.influence_candidates - len(top)), len(remaining))
        random_part = np.random.default_rng(args.seed).choice(remaining, count, replace=False)
        candidates = np.concatenate([top, random_part])
        candidate_loader = DataLoader(Subset(dataset, candidates.tolist()), batch_size=32)
        hessian_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        model = resnet18_cifar()
        model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu"))
        values = self_influence(
            model,
            torch.nn.CrossEntropyLoss(),
            candidate_loader,
            hessian_loader,
            InfluenceConfig(last_layers_only=True),
            pick_device(args.device),
        )
        influence_full[candidates] = values

    detectors = {"forgetting": dynamics}
    valid_influence = np.isfinite(influence_full)
    if valid_influence.any():
        filled = np.full(len(noise), np.nanmin(influence_full[valid_influence]))
        filled[valid_influence] = influence_full[valid_influence]
        detectors["influence"] = rank01(filled)
        detectors["combined"] = (rank01(dynamics) + rank01(filled)) / 2
    k = int(noise.sum())
    print("| detector | AP | precision@k |")
    print("|---|---:|---:|")
    figure = go.Figure()
    for name, score in detectors.items():
        ap = average_precision_score(noise, score)
        precision_at_k = float(noise[np.argsort(score)[::-1][:k]].mean())
        print(f"| {name} | {ap:.4f} | {precision_at_k:.4f} |")
        precision, recall, _ = precision_recall_curve(noise, score)
        figure.add_trace(go.Scatter(x=recall, y=precision, name=name))
    figure.update_layout(template="plotly_white", xaxis_title="recall", yaxis_title="precision")
    figure.write_html(run_dir / "pr_curves.html", include_plotlyjs="inline")
    raw = CIFAR10(args.data_dir, train=True, download=True)
    subset_indices = np.load(run_dir / "subset_indices.npy")
    extras = {
        "images": lambda index: np.asarray(raw[int(subset_indices[index])][0]),
        "image_labels": lambda index: "injected noise" if noise[index] else "clean",
    }
    if valid_influence.any():
        extras["influence"] = influence_full
    build_report(run_dir, run_dir / "report.html", extras)


if __name__ == "__main__":
    main()
