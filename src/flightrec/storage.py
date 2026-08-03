"""Crash-safe on-disk storage for recorded runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunData:
    """In-memory representation of a FlightRec run."""

    meta: dict[str, Any]
    scalars: dict[str, np.ndarray]
    correct: np.ndarray | None = None
    margin: np.ndarray | None = None
    spectrum_steps: np.ndarray | None = None
    eigs_high: np.ndarray | None = None
    eigs_low: np.ndarray | None = None


class RunWriter:
    """Append run records and atomically rewrite compact array snapshots."""

    def __init__(
        self, run_dir: str | Path, num_samples: int | None, config: dict[str, Any]
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.config = config
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._stream = (self.run_dir / "scalars.jsonl").open("a", encoding="utf-8")
        self._correct: list[np.ndarray] = []
        self._margin: list[np.ndarray] = []
        self._spectrum_steps: list[int] = []
        self._eigs_low: list[np.ndarray] = []
        self._eigs_high: list[np.ndarray] = []
        self.steps = 0

    def append_scalar(self, record: dict[str, Any]) -> None:
        """Immediately append a JSON record for crash safety."""
        # Python's JSON reader accepts NaN/Infinity; retaining them is more useful for an
        # instability recorder than raising while a training run is already failing.
        self._stream.write(json.dumps(record) + "\n")
        if record.get("kind") == "step":
            self.steps += 1

    def append_epoch(self, correct: np.ndarray, margin: np.ndarray) -> None:
        """Append and persist one epoch of per-example observations."""
        self._correct.append(np.asarray(correct, dtype=np.uint8).copy())
        self._margin.append(np.asarray(margin, dtype=np.float16).copy())
        self._write_per_example()
        self.flush()

    def append_spectrum(self, step: int, low: np.ndarray, high: np.ndarray) -> None:
        """Append one Hessian spectrum result."""
        self._spectrum_steps.append(int(step))
        self._eigs_low.append(np.asarray(low, dtype=np.float64))
        self._eigs_high.append(np.asarray(high, dtype=np.float64))
        self._write_spectra()

    def flush(self) -> None:
        """Flush the scalar stream."""
        self._stream.flush()

    def close(self, num_epochs: int) -> None:
        """Persist arrays and metadata, then close the scalar stream."""
        self._write_per_example()
        self._write_spectra()
        self.flush()
        self._stream.close()
        meta = {
            "created_at": self.created_at,
            "num_samples": self.num_samples,
            "num_epochs": int(num_epochs),
            "steps": int(self.steps),
            "config": self.config,
            "flightrec_version": "0.1.0",
        }
        self._atomic_json(self.run_dir / "meta.json", meta)

    def _write_per_example(self) -> None:
        if self.num_samples is None:
            return
        correct = (
            np.stack(self._correct) if self._correct else np.empty((0, self.num_samples), np.uint8)
        )
        margin = (
            np.stack(self._margin) if self._margin else np.empty((0, self.num_samples), np.float16)
        )
        self._atomic_npz(self.run_dir / "per_example.npz", correct=correct, margin=margin)

    def _write_spectra(self) -> None:
        if not self._spectrum_steps:
            return
        self._atomic_npz(
            self.run_dir / "spectra.npz",
            steps=np.asarray(self._spectrum_steps, dtype=np.int64),
            eigs_low=np.stack(self._eigs_low),
            eigs_high=np.stack(self._eigs_high),
        )

    @staticmethod
    def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)


class RunReader:
    """Read complete or interrupted FlightRec runs."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def read(self) -> RunData:
        """Load all files that are present in the run directory."""
        meta_path = self.run_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        records = self._read_jsonl(self.run_dir / "scalars.jsonl")
        scalars = self._records_to_arrays(records)
        correct = margin = None
        per_example = self.run_dir / "per_example.npz"
        if per_example.exists():
            with np.load(per_example) as data:
                correct, margin = data["correct"].copy(), data["margin"].copy()
        steps = low = high = None
        spectra = self.run_dir / "spectra.npz"
        if spectra.exists():
            with np.load(spectra) as data:
                steps = data["steps"].copy()
                low, high = data["eigs_low"].copy(), data["eigs_high"].copy()
        return RunData(meta, scalars, correct, margin, steps, high, low)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # ignore a final partial write after a crash
        return records

    @staticmethod
    def _records_to_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        keys = sorted({key for record in records for key in record})
        result: dict[str, np.ndarray] = {}
        for key in keys:
            values = [record.get(key, np.nan) for record in records]
            if key == "kind":
                result[key] = np.asarray(values, dtype=str)
            else:
                try:
                    result[key] = np.asarray(values, dtype=np.float64)
                except (TypeError, ValueError):
                    result[key] = np.asarray(values, dtype=object)
        return result


def read_run(run_dir: str | Path) -> RunData:
    """Load a run directory into a :class:`RunData`."""
    return RunReader(run_dir).read()
