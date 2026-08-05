import json

import numpy as np

from flightrec.storage import RunWriter, read_run


def test_storage_round_trip(tmp_path):
    writer = RunWriter(tmp_path, 3, {"probe_device": "cpu"})
    writer.append_scalar({"kind": "step", "step": 0, "loss": 1.25})
    writer.append_scalar({"kind": "eval", "step": 1, "test_acc": 0.75})
    correct = np.array([1, 0, 255], dtype=np.uint8)
    margin = np.array([2.0, -1.0, np.nan], dtype=np.float16)
    writer.append_epoch(correct, margin)
    writer.append_spectrum(1, np.array([-2.0]), np.array([3.0]))
    writer.close(1)
    run = read_run(tmp_path)
    np.testing.assert_array_equal(run.correct, correct[None])
    np.testing.assert_equal(run.margin, margin[None])
    np.testing.assert_array_equal(run.scalars["kind"], ["step", "eval"])
    np.testing.assert_allclose(run.scalars["loss"], [1.25, np.nan], equal_nan=True)
    np.testing.assert_allclose(run.scalars["test_acc"], [np.nan, 0.75], equal_nan=True)
    np.testing.assert_array_equal(run.spectrum_steps, [1])
    np.testing.assert_allclose(run.eigs_low, [[-2.0]])
    np.testing.assert_allclose(run.eigs_high, [[3.0]])
    assert run.meta["steps"] == 1


def test_crash_safe_jsonl_ignores_partial_final_line(tmp_path):
    path = tmp_path / "scalars.jsonl"
    path.write_text(json.dumps({"kind": "step", "step": 0, "loss": 2.0}) + "\n{")
    run = read_run(tmp_path)
    np.testing.assert_allclose(run.scalars["loss"], [2.0])
