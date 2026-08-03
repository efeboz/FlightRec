"""Train CIFAR-10 with controlled label noise and record learning dynamics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from model import resnet18_cifar
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from flightrec import FlightRecorder, IndexedDataset
from flightrec.utils import pick_device, seed_everything


class NoisySubset(Dataset[tuple[Tensor, int]]):
    """CIFAR subset with an explicit, reproducibly corrupted target array."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, labels: np.ndarray) -> None:
        self.dataset, self.indices, self.labels = dataset, indices, labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, _ = self.dataset[int(self.indices[index])]
        return image, int(self.labels[index])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs/cs1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subset", type=int)
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
        ]
    )
    train_base = datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(len(train_base))
    indices = all_indices[: args.subset] if args.subset else np.arange(len(train_base))
    original = np.asarray(train_base.targets, dtype=np.int64)[indices]
    labels = original.copy()
    noise_mask = np.zeros(len(indices), dtype=bool)
    noise_count = int(round(args.noise_rate * len(indices)))
    noisy = rng.choice(len(indices), noise_count, replace=False)
    noise_mask[noisy] = True
    offsets = rng.integers(1, 10, size=noise_count)
    labels[noisy] = (labels[noisy] + offsets) % 10
    np.save(run_dir / "noise_mask.npy", noise_mask)
    np.save(run_dir / "original_labels.npy", original)
    np.save(run_dir / "subset_indices.npy", indices)
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    train_set = IndexedDataset(NoisySubset(train_base, indices, labels))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2)
    model = resnet18_cifar().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    with FlightRecorder(model, run_dir, num_samples=len(train_set)) as recorder:
        for _ in range(args.epochs):
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
    torch.save(model.state_dict(), run_dir / "final_model.pt")


if __name__ == "__main__":
    main()
