"""Analyze noisy-label recovery and produce the CS1 report."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from flightrec import read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.influence import InfluenceConfig, influence_on
from flightrec.report import build_report
from flightrec.utils import pick_device, seed_everything

try:
    from .model import resnet18_cifar
except ImportError:
    from model import resnet18_cifar

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class LabelledSubset(Dataset[tuple[Tensor, int]]):
    """Present saved labels over stable indices into a transformed CIFAR dataset."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, labels: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, _ = self.dataset[int(self.indices[index])]
        return image, int(self.labels[index])


def parse_args() -> argparse.Namespace:
    """Parse analysis options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/cs1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--influence-candidates", type=int, default=5000)
    parser.add_argument("--influence-damping", type=float, default=0.01)
    parser.add_argument("--influence-cg-maxiter", type=int, default=100)
    parser.add_argument("--influence-hessian-batches", type=int, default=8)
    parser.add_argument("--influence-batch-size", type=int, default=128)
    parser.add_argument("--influence-reference-size", type=int, default=512)
    parser.add_argument("--skip-influence", action="store_true")
    return parser.parse_args()


def rank01(values: np.ndarray) -> np.ndarray:
    """Map values to average ranks in [0, 1], preserving ties."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    average_ranks = starts + (counts - 1) / 2.0
    ranks = average_ranks[inverse]
    return ranks / max(1, len(values) - 1)


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> TensorDataset:
    """Cache final-block features so head-only Hessian products stay inexpensive."""
    was_training = model.training
    model.eval()
    features: list[Tensor] = []
    targets: list[Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, labels = batch[:2]
        features.append(model.forward_features(inputs.to(device)).cpu())
        targets.append(labels.cpu())
    model.train(was_training)
    if not features:
        raise ValueError("feature loader yielded no batches")
    return TensorDataset(torch.cat(features), torch.cat(targets))


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
    influence_metadata: dict[str, object] | None = None

    if not args.skip_influence:
        influence_started = time.perf_counter()
        subset_indices = np.load(run_dir / "subset_indices.npy")
        noisy_labels = np.load(run_dir / "noisy_labels.npy")
        transform = Compose([ToTensor(), Normalize(CIFAR_MEAN, CIFAR_STD)])
        base = CIFAR10(args.data_dir, train=True, download=True, transform=transform)
        test_base = CIFAR10(args.data_dir, train=False, download=True, transform=transform)
        dataset = LabelledSubset(base, subset_indices, noisy_labels)
        top_count = min(2000, args.influence_candidates, len(noise))
        top = np.argsort(dynamics)[::-1][:top_count]
        remaining = np.setdiff1d(np.arange(len(noise)), top)
        count = min(max(0, args.influence_candidates - len(top)), len(remaining))
        random_part = np.random.default_rng(args.seed).choice(remaining, count, replace=False)
        candidates = np.concatenate([top, random_part])
        generator = torch.Generator().manual_seed(args.seed)
        candidate_image_loader = DataLoader(
            Subset(dataset, candidates.tolist()), batch_size=args.influence_batch_size
        )
        hessian_image_loader = DataLoader(
            dataset, batch_size=128, shuffle=True, generator=generator
        )
        reference_image_loader = DataLoader(
            test_base, batch_size=args.influence_reference_size, shuffle=False
        )
        model = resnet18_cifar()
        model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu"))
        feature_device = pick_device(args.device)
        model.to(feature_device)
        candidate_features = extract_features(model, candidate_image_loader, feature_device)
        hessian_features = extract_features(
            model,
            hessian_image_loader,
            feature_device,
            max_batches=args.influence_hessian_batches,
        )
        reference_features = extract_features(
            model, reference_image_loader, feature_device, max_batches=1
        )
        classifier = torch.nn.Linear(model.fc.in_features, model.fc.out_features)
        classifier.load_state_dict(model.fc.cpu().state_dict())
        values = influence_on(
            classifier,
            torch.nn.CrossEntropyLoss(),
            reference_features.tensors,
            DataLoader(candidate_features, batch_size=args.influence_batch_size),
            DataLoader(hessian_features, batch_size=128),
            InfluenceConfig(
                damping=args.influence_damping,
                cg_maxiter=args.influence_cg_maxiter,
                hessian_batches=args.influence_hessian_batches,
                last_layers_only=False,
            ),
            "cpu",
        )
        influence_full[candidates] = values
        influence_runtime = time.perf_counter() - influence_started
        influence_metadata = {
            "method": "shared validation influence on cached final-head features",
            "candidate_count": int(len(candidates)),
            "flagged_candidate_count": int(len(top)),
            "random_candidate_count": int(len(random_part)),
            "reference_count": int(len(reference_features)),
            "hessian_count": int(len(hessian_features)),
            "runtime_seconds": influence_runtime,
            "damping": args.influence_damping,
            "cg_maxiter": args.influence_cg_maxiter,
        }
        np.savez_compressed(run_dir / "influence_results.npz", indices=candidates, scores=values)
        (run_dir / "influence_metadata.json").write_text(json.dumps(influence_metadata, indent=2))

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
    measured: dict[str, dict[str, float]] = {}
    for name, score in detectors.items():
        ap = average_precision_score(noise, score)
        precision_at_k = float(noise[np.argsort(score)[::-1][:k]].mean())
        measured[name] = {"average_precision": ap, "precision_at_k": precision_at_k}
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
        extras["influence_label"] = "shared validation influence"
    build_report(run_dir, run_dir / "report.html", extras)
    (run_dir / "analysis_metrics.json").write_text(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
