import gc
import warnings

import numpy as np
import torch
from torch import nn

from flightrec import FlightRecorder, read_run


def test_recorder_tracks_last_observation_and_scalars(tmp_path):
    model = nn.Linear(2, 2)
    recorder = FlightRecorder(model, tmp_path, num_samples=3)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    for indices in (torch.tensor([0, 1]), torch.tensor([0, 2])):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        model.zero_grad()
        loss.backward()
        recorder.record_step(
            loss=loss,
            logits=logits,
            targets=targets,
            sample_indices=indices,
            lr=0.1,
        )
    expected = np.full(3, 255, np.uint8)
    expected[[0, 2]] = (logits.argmax(1) == targets).numpy()
    expected[1] = (model(inputs).argmax(1) == targets).numpy()[1]
    recorder.record_eval(test_acc=0.5)
    recorder.epoch_end()
    recorder.close()
    run = read_run(tmp_path)
    np.testing.assert_array_equal(run.correct[0], expected)
    assert run.meta["steps"] == 2
    assert set(run.scalars["kind"]) == {"step", "eval"}
    step = run.scalars["kind"] == "step"
    expected_grad_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    ).item()
    expected_param_norm = torch.linalg.vector_norm(
        torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    ).item()
    np.testing.assert_allclose(run.scalars["grad_norm"][step].astype(float), expected_grad_norm)
    np.testing.assert_allclose(run.scalars["param_norm"][step].astype(float), expected_param_norm)


def test_recorder_does_not_retain_graph_tensors(tmp_path):
    model = nn.Linear(4, 3)
    recorder = FlightRecorder(model, tmp_path)
    inputs = torch.randn(8, 4)
    targets = torch.randint(3, (8,))
    counts = []
    for step in range(100):
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        recorder.record_step(loss=loss)
        if step in (19, 99):
            del logits, loss
            gc.collect()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                counts.append(sum(isinstance(obj, torch.Tensor) for obj in gc.get_objects()))
    recorder.close()
    assert counts[1] <= counts[0] + 4


def test_interrupted_recorder_has_readable_scalars(tmp_path):
    model = nn.Linear(1, 1)
    recorder = FlightRecorder(model, tmp_path)
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    recorder.record_step(loss=loss)
    recorder.epoch_end()
    assert read_run(tmp_path).scalars["step"].tolist() == [0.0]
    recorder.close()


def test_spectrum_probe_leaves_training_model_untouched(tmp_path):
    torch.manual_seed(11)
    model = nn.Linear(2, 2)
    model.train()
    inputs = torch.randn(5, 2)
    targets = torch.randint(0, 2, (5,))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def probe_loss(probe_model, batch):
        probe_inputs, probe_targets = batch
        return nn.functional.cross_entropy(probe_model(probe_inputs), probe_targets)

    recorder = FlightRecorder(
        model,
        tmp_path,
        spectrum_every=1,
        spectrum_k=1,
        probe_device="cpu",
        probe_loss_fn=probe_loss,
        probe_batch=(inputs, targets),
    )
    loss = nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    recorder.record_step(loss=loss)
    recorder.close()

    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    run = read_run(tmp_path)
    assert run.spectrum_steps is not None
    assert run.spectrum_steps.tolist() == [1]
