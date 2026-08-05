"""Train CIFAR-10 with controlled label noise and record learning dynamics."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from flightrec import FlightRecorder, IndexedDataset
from flightrec.utils import pick_device, seed_everything

try:
    from .model import resnet18_cifar
except ImportError:
    from model import resnet18_cifar

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class NoisySubset(Dataset[tuple[Tensor, int]]):
    """CIFAR subset with an explicit, reproducibly corrupted target array."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, labels: np.ndarray) -> None:
        self.dataset, self.indices, self.labels = dataset, indices, labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, _ = self.dataset[int(self.indices[index])]
        return image, int(self.labels[index])


def inject_label_noise(
    original: np.ndarray, noise_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return corrupted labels and the exact boolean corruption mask."""
    labels = np.asarray(original, dtype=np.int64).copy()
    noise_mask = np.zeros(len(labels), dtype=bool)
    noise_count = int(round(noise_rate * len(labels)))
    noisy = rng.choice(len(labels), noise_count, replace=False)
    noise_mask[noisy] = True
    offsets = rng.integers(1, 10, size=noise_count)
    labels[noisy] = (labels[noisy] + offsets) % 10
    return labels, noise_mask


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        help="cosine-schedule horizon; defaults to --epochs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs/cs1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-batch-size", type=int, default=512)
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate mean cross-entropy and accuracy."""
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = correct = count = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += float(criterion(logits, targets).item())
            correct += float((logits.argmax(1) == targets).sum().item())
            count += len(targets)
    model.train()
    return loss / count, correct / count


def main() -> None:
    """Run the noisy-label training experiment."""
    started = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    train_base = datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=train_transform
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(len(train_base))
    indices = all_indices[: args.subset] if args.subset else np.arange(len(train_base))
    original = np.asarray(train_base.targets, dtype=np.int64)[indices]
    labels, noise_mask = inject_label_noise(original, args.noise_rate, rng)
    np.save(run_dir / "noise_mask.npy", noise_mask)
    np.save(run_dir / "original_labels.npy", original)
    np.save(run_dir / "noisy_labels.npy", labels)
    np.save(run_dir / "subset_indices.npy", indices)
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    train_set = IndexedDataset(NoisySubset(train_base, indices, labels))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_set, batch_size=args.test_batch_size, num_workers=args.num_workers
    )
    model = resnet18_cifar().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule_epochs = args.schedule_epochs or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule_epochs)
    with FlightRecorder(model, run_dir, num_samples=len(train_set)) as recorder:
        for epoch in range(args.epochs):
            for inputs, targets, indices_batch in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                recorder.record_step(
                    loss=loss,
                    logits=logits,
                    targets=targets,
                    sample_indices=indices_batch,
                    lr=optimizer.param_groups[0]["lr"],
                )
                optimizer.step()
            test_loss, test_acc = evaluate(model, test_loader, device)
            recorder.record_eval(test_loss=test_loss, test_acc=test_acc)
            recorder.epoch_end()
            scheduler.step()
            print(
                f"epoch {epoch + 1}/{args.epochs}: "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}",
                flush=True,
            )
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    environment = {
        "train_runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    (run_dir / "run_environment.json").write_text(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
