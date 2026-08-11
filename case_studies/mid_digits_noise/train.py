"""Train a small digit classifier on deliberately corrupted labels and record everything.

This is the mid-sized case study: the complete CS1 workflow -- injected label noise, per-example
dynamics, periodic curvature probes, influence, and an illustrated report -- on the 8x8 digit set
bundled with scikit-learn, so it runs in minutes on a laptop and downloads nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec import FlightRecorder, IndexedDataset
from flightrec.utils import pick_device, seed_everything

PIXEL_MAX = 16.0


class DigitCNN(nn.Module):
    """A compact convolutional classifier for 8x8 grayscale digits."""

    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Linear(2 * width * 4 * 4, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits."""
        return self.head(self.features(inputs))


def inject_label_noise(
    original: np.ndarray, noise_rate: float, rng: np.random.Generator, num_classes: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Reassign a fraction of labels uniformly among the other classes.

    Returns the corrupted labels and the exact boolean mask of which entries changed, matching
    the corruption semantics of the CIFAR-10 case study.
    """
    labels = np.asarray(original, dtype=np.int64).copy()
    mask = np.zeros(len(labels), dtype=bool)
    count = int(round(noise_rate * len(labels)))
    chosen = rng.choice(len(labels), count, replace=False)
    mask[chosen] = True
    labels[chosen] = (labels[chosen] + rng.integers(1, num_classes, size=count)) % num_classes
    return labels, mask


def load_digit_data(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a stratified train/test split of the bundled digit images in NCHW form."""
    digits = load_digits()
    images = (digits.images / PIXEL_MAX).astype(np.float32)[:, None, :, :]
    train_x, test_x, train_y, test_y = train_test_split(
        images, digits.target, test_size=0.2, stratify=digits.target, random_state=seed
    )
    return (
        torch.tensor(train_x),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(test_x),
        torch.tensor(test_y, dtype=torch.long),
    )


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/mid-digits")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--noise-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--spectrum-every", type=int, default=150)
    parser.add_argument("--spectrum-k", type=int, default=2)
    parser.add_argument("--spectrum-lanczos-steps", type=int, default=16)
    parser.add_argument("--no-spectrum", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, features: Tensor, labels: Tensor) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy in evaluation mode."""
    was_training = model.training
    model.eval()
    logits = model(features)
    loss = float(nn.functional.cross_entropy(logits, labels).item())
    accuracy = float((logits.argmax(1) == labels).float().mean().item())
    model.train(was_training)
    return loss, accuracy


def main() -> None:
    """Train the noisy-label digit model under full instrumentation."""
    args = parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    device = pick_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "scalars.jsonl").exists():
        raise FileExistsError(f"{run_dir} already contains a run; choose a fresh --run-dir")

    train_x, clean_y, test_x, test_y = load_digit_data(args.seed)
    noisy_y, noise_mask = inject_label_noise(
        clean_y.numpy(), args.noise_rate, np.random.default_rng(args.seed)
    )
    np.save(run_dir / "noise_mask.npy", noise_mask)
    np.save(run_dir / "original_labels.npy", clean_y.numpy())
    np.save(run_dir / "noisy_labels.npy", noisy_y)
    np.save(run_dir / "train_images.npy", train_x.numpy())
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    train_y = torch.tensor(noisy_y, dtype=torch.long)
    train_set = TensorDataset(train_x, train_y)
    loader = DataLoader(
        IndexedDataset(train_set), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    probe_batch = (train_x[:256].cpu(), train_y[:256].cpu())

    def probe_loss(probe_model: nn.Module, batch: object) -> Tensor:
        inputs, targets = batch
        return criterion(probe_model(inputs), targets)

    with FlightRecorder(
        model,
        run_dir,
        num_samples=len(train_set),
        spectrum_every=None if args.no_spectrum else args.spectrum_every,
        spectrum_k=args.spectrum_k,
        spectrum_lanczos_steps=args.spectrum_lanczos_steps,
        probe_device="cpu",
        probe_loss_fn=probe_loss,
        probe_batch=probe_batch,
    ) as recorder:
        for epoch in range(args.epochs):
            for inputs, targets, indices in loader:
                inputs, targets = inputs.to(device), targets.to(device)
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
            test_loss, test_acc = evaluate(model, test_x, test_y)
            train_loss, train_acc = evaluate(model, train_x, train_y)
            recorder.record_eval(
                test_loss=test_loss,
                test_acc=test_acc,
                train_eval_loss=train_loss,
                train_acc=train_acc,
            )
            recorder.epoch_end()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs}: train_acc={train_acc:.4f} "
                    f"test_acc={test_acc:.4f}",
                    flush=True,
                )
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    runtime = time.perf_counter() - started
    (run_dir / "run_environment.json").write_text(
        json.dumps(
            {
                "train_runtime_seconds": runtime,
                "device": str(device),
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
            },
            indent=2,
        )
    )
    print(f"training finished in {runtime:.2f} s; run directory {run_dir}")


if __name__ == "__main__":
    main()
