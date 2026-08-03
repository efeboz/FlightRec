"""Measure FlightRecorder overhead on a small CIFAR-shaped training loop."""

import argparse
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from flightrec import FlightRecorder
from flightrec.utils import seed_everything


def trial(mode: str, steps: int) -> float:
    """Time one benchmark mode."""
    # Convolutional work makes this representative of CS1 rather than a recorder-bound
    # linear microbenchmark.
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(128, 3, 32, 32)
    targets = torch.randint(10, (128,))
    indices = torch.arange(128)
    recorder = None
    temporary = tempfile.TemporaryDirectory()
    if mode != "off":
        recorder = FlightRecorder(
            model, Path(temporary.name), num_samples=128 if mode == "full" else None
        )
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        if recorder is not None:
            recorder.record_step(
                loss=loss,
                logits=logits if mode == "full" else None,
                targets=targets if mode == "full" else None,
                sample_indices=indices if mode == "full" else None,
                lr=0.01,
            )
        optimizer.step()
    elapsed = time.perf_counter() - start
    if recorder is not None:
        recorder.epoch_end()
        recorder.close()
    temporary.cleanup()
    return elapsed


def main() -> None:
    """Print recorder overhead relative to an uninstrumented loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    values = {mode: trial(mode, args.steps) for mode in ("off", "cheap", "full")}
    print("| mode | seconds | overhead |")
    print("|---|---:|---:|")
    for mode, elapsed in values.items():
        print(f"| {mode} | {elapsed:.3f} | {(elapsed / values['off'] - 1) * 100:.2f}% |")


if __name__ == "__main__":
    main()
