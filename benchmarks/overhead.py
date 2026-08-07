"""Measure FlightRecorder overhead on the CS1 ResNet training loop."""

import argparse
import json
import platform
import statistics
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

MODES = ("off", "cheap", "full")


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


def measure(steps: int, seed: int, repeats: int) -> dict[str, list[float]]:
    """Time every mode once per repeat, rotating the order to cancel drift.

    A single pass cannot separate a one-percent recorder cost from machine noise, so each
    repeat runs all modes and rotates which mode goes first. Position within a repeat is
    therefore balanced across modes rather than always favouring the same one.
    """
    samples: dict[str, list[float]] = {mode: [] for mode in MODES}
    for repeat in range(repeats):
        offset = repeat % len(MODES)
        for mode in MODES[offset:] + MODES[:offset]:
            samples[mode].append(trial(mode, steps, seed))
    return samples


def main() -> None:
    """Print recorder overhead relative to an uninstrumented loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    seed_everything(args.seed)
    samples = measure(args.steps, args.seed, args.repeats)
    median = {mode: statistics.median(values) for mode, values in samples.items()}
    fastest = {mode: min(values) for mode, values in samples.items()}
    print(f"{args.steps} steps, {args.repeats} repeats, CS1 ResNet at batch 128")
    print("| mode | median s | fastest s | median overhead | fastest overhead |")
    print("|---|---:|---:|---:|---:|")
    for mode in MODES:
        print(
            f"| {mode} | {median[mode]:.3f} | {fastest[mode]:.3f} "
            f"| {(median[mode] / median['off'] - 1) * 100:+.2f}% "
            f"| {(fastest[mode] / fastest['off'] - 1) * 100:+.2f}% |"
        )
    if args.json_out:
        result = {
            "steps": args.steps,
            "seed": args.seed,
            "repeats": args.repeats,
            "model": "CS1 CifarResNet",
            "batch_size": 128,
            "seconds_per_repeat": samples,
            "seconds_median": median,
            "seconds_fastest": fastest,
            "overhead_percent_median": {
                mode: (value / median["off"] - 1.0) * 100.0 for mode, value in median.items()
            },
            "overhead_percent_fastest": {
                mode: (value / fastest["off"] - 1.0) * 100.0 for mode, value in fastest.items()
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
