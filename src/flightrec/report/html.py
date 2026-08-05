"""Compose a standalone FlightRec post-mortem report."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from flightrec.analysis.events import compute_event_stats, suspicion_score
from flightrec.analysis.phases import detect_phases
from flightrec.report.figures import (
    first_learned_figure,
    forgetting_figure,
    influence_figure,
    margin_forgetting_figure,
    spectrum_figure,
    timeline_figure,
)
from flightrec.storage import read_run


def _table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> str:
    heading = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def _thumbnail(
    image_fn: Callable[[int], np.ndarray], index: int, label_fn: Callable[[int], str] | None
) -> str:
    from PIL import Image

    array = np.asarray(image_fn(index))
    if array.dtype != np.uint8:
        array = np.clip(array * (255 if array.max(initial=0) <= 1 else 1), 0, 255).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((64, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    caption = str(index) if label_fn is None else f"{index} · {html.escape(str(label_fn(index)))}"
    return (
        f'<figure><img src="data:image/png;base64,{encoded}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def build_report(
    run_dir: str | Path,
    out_path: str | Path,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Build one self-contained HTML report and return its output path."""
    extras = extras or {}
    run = read_run(run_dir)
    phases = detect_phases(run)
    figures = [timeline_figure(run, phases)]
    if run.spectrum_steps is not None:
        figures.append(spectrum_figure(run, phases))

    sections = ["<h1>FlightRec post-mortem</h1>"]
    meta_rows = [(key, value) for key, value in sorted(run.meta.items()) if key != "config"]
    for name, values in sorted(run.scalars.items()):
        if name in {"kind", "step", "epoch", "wall_time"}:
            continue
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError):
            continue
        valid = numeric[np.isfinite(numeric)]
        if len(valid):
            meta_rows.append((f"final {name}", f"{valid[-1]:.6g}"))
    sections.extend(["<h2>Run summary</h2>", _table(meta_rows, ("field", "value"))])

    stats = None
    scores = None
    if run.correct is not None:
        stats = compute_event_stats(run)
        scores = suspicion_score(stats)
        figures.extend(
            [forgetting_figure(stats), first_learned_figure(stats), margin_forgetting_figure(stats)]
        )
        order = np.argsort(scores)[::-1][:50]
        assert run.margin is not None
        observed_margin = np.isfinite(run.margin)
        epoch_indices = np.broadcast_to(np.arange(len(run.margin))[:, None], run.margin.shape)
        last_margin_epoch = np.max(np.where(observed_margin, epoch_indices, -1), axis=0)
        final_margin = np.full(run.margin.shape[1], np.nan, dtype=float)
        seen = last_margin_epoch >= 0
        final_margin[seen] = run.margin[last_margin_epoch[seen], np.flatnonzero(seen)]
        rows = [
            (
                int(index),
                f"{scores[index]:.4f}",
                int(stats.forgetting_count[index]),
                int(stats.first_learned[index]),
                f"{final_margin[index]:.4g}",
            )
            for index in order
        ]
        sections.extend(
            [
                "<h2>Flagged samples</h2>",
                _table(rows, ("index", "score", "forgotten", "first learned", "final margin")),
            ]
        )
        image_fn = extras.get("images")
        if callable(image_fn):
            label_fn = extras.get("image_labels")
            if not callable(label_fn):
                label_fn = None
            sections.append(
                '<div class="gallery">'
                + "".join(_thumbnail(image_fn, int(i), label_fn) for i in order)
                + "</div>"
            )

    influence = extras.get("influence")
    if influence is not None and scores is not None:
        influence_array = np.asarray(influence, dtype=float)
        size = min(len(scores), len(influence_array))
        valid = np.isfinite(influence_array[:size]) & np.isfinite(scores[:size])
        if valid.sum() >= 2:
            rho = float(spearmanr(scores[:size][valid], influence_array[:size][valid]).statistic)
            figures.append(
                influence_figure(scores[:size][valid], influence_array[:size][valid], rho)
            )

    plot_html = []
    for index, figure in enumerate(figures):
        plot_html.append(
            figure.to_html(full_html=False, include_plotlyjs="inline" if index == 0 else False)
        )
    sections.insert(3, "<h2>Timeline and diagnostics</h2>" + "".join(plot_html))
    document = (
        """<!doctype html><html><head><meta charset="utf-8"><title>FlightRec report</title>
<style>body{font-family:system-ui;margin:2rem;max-width:1200px}table{border-collapse:collapse}
th,td{padding:.35rem .65rem;border:1px solid #ddd;text-align:left}
.gallery{display:flex;flex-wrap:wrap}
figure{margin:.4rem;text-align:center}img{image-rendering:auto}</style></head><body>"""
        + "".join(sections)
        + "</body></html>"
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
