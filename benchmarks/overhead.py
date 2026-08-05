"""Measure FlightRecorder overhead on the CS1 ResNet training loop."""

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from flightrec import FlightRecorder
from flightrec.utils import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from case_studies.cs1_cifar_label_noise.model import resnet18_cifar  # noqa: E402


def trial(mode: str, steps: int, seed: int) -> float:
    """Time one mode using the exact CS1 model, batch size, and optimizer."""
    seed_everything(seed)
    model = resnet18_cifar()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    inputs = torch.randn(128, 3, 32, 32)
    targets = torch.randint(10, (128,))
    indices = torch.arange(128)
    recorder = None
    temporary = tempfile.TemporaryDirectory()
    if mode != "off":
        recorder = FlightRecorder(
            model, Path(temporary.name), num_samples=128 if mode == "full" else None
        )

    def step() -> None:
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
                lr=0.1,
            )
        optimizer.step()

    for _ in range(min(5, steps)):
        step()
    start = time.perf_counter()
    for _ in range(steps):
        step()
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
    parser.add_argument("--json-out")
    args = parser.parse_args()
    seed_everything(args.seed)
    values = {mode: trial(mode, args.steps, args.seed) for mode in ("off", "cheap", "full")}
    print("| mode | seconds | overhead |")
    print("|---|---:|---:|")
    for mode, elapsed in values.items():
        print(f"| {mode} | {elapsed:.3f} | {(elapsed / values['off'] - 1) * 100:.2f}% |")
    if args.json_out:
        result = {
            "steps": args.steps,
            "seed": args.seed,
            "model": "CS1 CifarResNet",
            "batch_size": 128,
            "seconds": values,
            "overhead_percent": {
                mode: (elapsed / values["off"] - 1.0) * 100.0 for mode, elapsed in values.items()
            },
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        }
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
