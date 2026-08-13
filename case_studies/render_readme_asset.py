"""Render the README overview from two measured case-study runs."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10

from flightrec import read_run
from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.phases import detect_phases

WIDTH = 1200
HEIGHT = 420
INK = "#172033"
PHASE_COLORS = {
    "memorization": "#fff2c7",
    "generalization": "#d9f8df",
    "plateau": "#edf0f4",
    "instability": "#fde2e2",
    "fitting": "#deebff",
}


def parse_args() -> argparse.Namespace:
    """Parse input run directories and output location."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs1-run", required=True)
    parser.add_argument("--cs2-run", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="assets/overview.png")
    return parser.parse_args()


def gallery_indices(
    scores: np.ndarray, noise_mask: np.ndarray, count: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Return the highest-ranked overall and non-injected examples."""
    if scores.shape != noise_mask.shape:
        raise ValueError("scores and noise_mask must have the same shape")
    if count < 1:
        raise ValueError("count must be positive")
    order = np.argsort(scores, kind="stable")[::-1]
    overall = order[:count]
    not_injected = order[~noise_mask[order]][:count]
    if len(not_injected) < count:
        raise ValueError(f"need at least {count} non-injected examples")
    return overall, not_injected


def main() -> None:
    """Draw a real accuracy timeline beside real flagged CIFAR thumbnails."""
    args = parse_args()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((40, 22), "Delayed generalization timeline", fill=INK, font=font)
    draw.text((650, 22), "Flagged CIFAR-10 training samples", fill=INK, font=font)

    cs2 = read_run(args.cs2_run)
    phases = detect_phases(cs2)
    left, top, right, bottom = 55, 62, 575, 375
    draw.rectangle((left, top, right, bottom), fill="white", outline="#d9e0ea")
    maximum_step = max(1.0, float(np.nanmax(cs2.scalars["step"])))

    def x_position(step: float) -> float:
        return left + 12 + (right - left - 24) * np.log10(max(1.0, step)) / np.log10(maximum_step)

    def y_position(value: float) -> float:
        return bottom - 15 - (bottom - top - 30) * np.clip(value, 0.0, 1.0)

    for phase in phases.phases:
        x0, x1 = x_position(phase.start_step), x_position(phase.end_step)
        draw.rectangle((x0, top + 1, x1, bottom - 1), fill=PHASE_COLORS.get(phase.label, "#f3f4f6"))
    kinds = cs2.scalars["kind"]
    evaluation = kinds == "eval"
    steps = cs2.scalars["step"][evaluation].astype(float)
    for field, color in (("train_acc", "#2563eb"), ("test_acc", "#16a34a")):
        values = cs2.scalars[field][evaluation].astype(float)
        points = [
            (x_position(step), y_position(value)) for step, value in zip(steps, values, strict=True)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    draw.text((left + 12, bottom - 12), "step (log scale)", fill=INK, font=font)
    draw.line((left + 340, top + 18, left + 365, top + 18), fill="#2563eb", width=3)
    draw.text((left + 370, top + 12), "train", fill=INK, font=font)
    draw.line((left + 430, top + 18, left + 455, top + 18), fill="#16a34a", width=3)
    draw.text((left + 460, top + 12), "test", fill=INK, font=font)

    cs1_root = Path(args.cs1_run)
    cs1 = read_run(cs1_root)
    scores = suspicion_score(compute_event_stats(cs1))
    subset_indices = np.load(cs1_root / "subset_indices.npy")
    noise = np.load(cs1_root / "noise_mask.npy")
    overall, not_injected = gallery_indices(scores, noise)
    raw = CIFAR10(args.data_dir, train=True, download=False)
    draw.text((650, 44), "Top 5 overall", fill=INK, font=font)
    draw.text((650, 209), "Top 5 among samples not artificially corrupted", fill=INK, font=font)
    for row, selected in enumerate((overall, not_injected)):
        for column, index in enumerate(selected):
            x = 650 + column * 105
            y = 66 + row * 165
            thumbnail = raw[int(subset_indices[index])][0].resize((92, 92))
            canvas.paste(thumbnail, (x, y))
            color = "#dc2626" if noise[index] else "#2563eb"
            draw.rectangle((x - 2, y - 2, x + 93, y + 93), outline=color, width=3)
            draw.text((x, y + 100), f"#{int(index)} score {scores[index]:.2f}", fill=INK, font=font)
            label = "injected noise" if noise[index] else "not injected"
            draw.text((x, y + 116), label, fill=color, font=font)
    draw.text(
        (650, 397),
        "Red: injected label noise   Blue: not injected (may still be hard or mislabeled)",
        fill=INK,
        font=font,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    print(output)


if __name__ == "__main__":
    main()
