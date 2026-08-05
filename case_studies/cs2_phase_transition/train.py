"""Record a modular-addition delayed generalization run."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flightrec import FlightRecorder, IndexedDataset
from flightrec.utils import pick_device, seed_everything

try:
    from .model import ModularAdditionTransformer
except ImportError:
    from model import ModularAdditionTransformer


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="runs/cs2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--spectrum-every", type=int, default=1000)
    parser.add_argument("--spectrum-k", type=int, default=5)
    parser.add_argument("--no-spectrum", action="store_true")
    return parser.parse_args()


def modular_data(modulus: int, train_frac: float, seed: int) -> tuple[TensorDataset, TensorDataset]:
    """Create a seeded split of all ordered modular-addition pairs."""
    a, b = torch.meshgrid(torch.arange(modulus), torch.arange(modulus), indexing="ij")
    inputs = torch.stack([a.flatten(), b.flatten()], dim=1)
    targets = (inputs[:, 0] + inputs[:, 1]) % modulus
    order = np.random.default_rng(seed).permutation(len(inputs))
    split = int(round(train_frac * len(inputs)))
    train_idx = torch.as_tensor(order[:split])
    test_idx = torch.as_tensor(order[split:])
    return TensorDataset(inputs[train_idx], targets[train_idx]), TensorDataset(
        inputs[test_idx], targets[test_idx]
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate loss and accuracy."""
    was_training = model.training
    model.eval()
    total_loss = total_correct = total = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets).item())
        total_correct += float((logits.argmax(1) == targets).sum().item())
        total += len(targets)
    model.train(was_training)
    return total_loss / total, total_correct / total


def main() -> None:
    """Train and record modular addition."""
    started = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    probe_device = torch.device("cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))
    train_set, test_set = modular_data(args.p, args.train_frac, args.seed)
    train_loader = DataLoader(
        IndexedDataset(train_set), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    train_eval_loader = DataLoader(train_set, batch_size=1024)
    test_loader = DataLoader(test_set, batch_size=1024)
    fixed_inputs = train_set.tensors[0][:512]
    fixed_targets = train_set.tensors[1][:512]
    model = ModularAdditionTransformer(args.p).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1.0, betas=(0.9, 0.98)
    )
    spectrum_every = None if args.no_spectrum else args.spectrum_every

    def probe_loss(probe_model: torch.nn.Module, batch: object) -> torch.Tensor:
        inputs, targets = batch
        return criterion(probe_model(inputs), targets)

    checkpoints: set[str] = set()
    loader_iterator = iter(train_loader)
    with FlightRecorder(
        model,
        run_dir,
        num_samples=len(train_set),
        spectrum_every=spectrum_every,
        spectrum_k=args.spectrum_k,
        probe_device=probe_device,
        probe_loss_fn=probe_loss,
        probe_batch=(fixed_inputs, fixed_targets),
    ) as recorder:
        for step in range(1, args.steps + 1):
            try:
                inputs, targets, indices = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(train_loader)
                inputs, targets, indices = next(loader_iterator)
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
            if step % args.eval_every == 0:
                test_loss, test_acc = evaluate(model, test_loader, device)
                train_loss, train_acc = evaluate(model, train_eval_loader, device)
                recorder.record_eval(
                    train_eval_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                )
                recorder.epoch_end()
                for name, threshold in (("pre", 0.1), ("mid", 0.5), ("post", 0.9)):
                    if name not in checkpoints and test_acc >= threshold:
                        torch.save(model.state_dict(), run_dir / f"checkpoint_{name}.pt")
                        checkpoints.add(name)
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
