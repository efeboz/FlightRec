import copy

import numpy as np
import pytest
import torch
from scipy.sparse.linalg import LinearOperator
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec.analysis.influence import (
    InfluenceConfig,
    _candidate_gradients,
    influence_on,
    self_influence,
)


def train_model(dataset, damping, state=None):
    model = nn.Linear(2, 2, bias=True).double()
    if state is not None:
        model.load_state_dict(copy.deepcopy(state))
    inputs, targets = dataset.tensors
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, tolerance_grad=1e-10)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss = loss + 0.5 * damping * sum((param**2).sum() for param in model.parameters())
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def test_influence_correlates_with_leave_one_out_retraining():
    features, labels = make_blobs(
        n_samples=80, centers=2, n_features=2, random_state=4, cluster_std=2.0
    )
    features = torch.tensor(features, dtype=torch.float64)
    labels = torch.tensor(labels, dtype=torch.long)
    train = TensorDataset(features[:70], labels[:70])
    test_batch = (features[70:], labels[70:])
    damping = 0.1
    model = train_model(train, damping)
    predicted = influence_on(
        model,
        nn.CrossEntropyLoss(),
        test_batch,
        DataLoader(train, batch_size=10, shuffle=False),
        DataLoader(train, batch_size=35, shuffle=False),
        InfluenceConfig(
            damping=damping, cg_tol=1e-8, cg_maxiter=100, hessian_batches=2, last_layers_only=False
        ),
        "cpu",
    )
    top = np.argsort(np.abs(predicted))[::-1][:10]
    baseline = float(nn.functional.cross_entropy(model(test_batch[0]), test_batch[1]).item())
    actual = []
    for removed in top:
        keep = torch.arange(len(train)) != int(removed)
        reduced = TensorDataset(features[:70][keep], labels[:70][keep])
        retrained = train_model(reduced, damping, model.state_dict())
        actual.append(
            float(nn.functional.cross_entropy(retrained(test_batch[0]), test_batch[1]).item())
            - baseline
        )
    # Removing a point is the negative of infinitesimal upweighting, up to a positive 1/n scale.
    correlation = spearmanr(-predicted[top], actual).statistic
    assert correlation > 0.8


def test_self_influence_matches_materialized_inverse_hessian():
    torch.manual_seed(7)
    inputs = torch.tensor([[-1.0, 0.5], [-0.4, 1.2], [0.7, -0.3], [1.1, 0.9]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(inputs, targets)
    model = nn.Linear(2, 2).double()
    flat = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    damping = 0.25

    def full_loss(vector):
        weight = vector[:4].reshape(2, 2)
        bias = vector[4:]
        return nn.functional.cross_entropy(inputs @ weight.T + bias, targets)

    hessian = torch.autograd.functional.hessian(full_loss, flat).numpy()
    system = hessian + damping * np.eye(len(flat))
    expected = []
    for sample, target in dataset:

        def one_loss(vector, sample=sample, target=target):
            weight = vector[:4].reshape(2, 2)
            bias = vector[4:]
            logits = sample[None] @ weight.T + bias
            return nn.functional.cross_entropy(logits, target[None])

        gradient = torch.autograd.functional.jacobian(one_loss, flat).numpy()
        expected.append(float(gradient @ np.linalg.solve(system, gradient)))

    actual = self_influence(
        model,
        nn.CrossEntropyLoss(),
        DataLoader(dataset, batch_size=2, shuffle=False),
        DataLoader(dataset, batch_size=4, shuffle=False),
        InfluenceConfig(
            damping=damping,
            cg_tol=1e-10,
            cg_maxiter=100,
            hessian_batches=1,
            last_layers_only=False,
        ),
        "cpu",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_self_influence_rejects_nonconverged_cg(monkeypatch):
    model = nn.Linear(1, 2).double()
    dataset = TensorDataset(torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.long))

    def fake_system(*_args, **_kwargs):
        params = list(model.parameters())
        size = sum(parameter.numel() for parameter in params)
        operator = LinearOperator((size, size), matvec=lambda vector: vector, dtype=np.float64)
        return params, operator

    def fake_cg(*_args, **_kwargs):
        return np.ones(4), 7

    monkeypatch.setattr("flightrec.analysis.influence._system", fake_system)
    monkeypatch.setattr("flightrec.analysis.influence.cg", fake_cg)
    with pytest.raises(RuntimeError, match="did not converge"):
        self_influence(
            model,
            nn.CrossEntropyLoss(),
            DataLoader(dataset),
            DataLoader(dataset),
            InfluenceConfig(last_layers_only=False),
            "cpu",
        )


def test_candidate_gradient_microbatch_reuses_forward_pass():
    class CountingLinear(nn.Linear):
        calls = 0

        def forward(self, inputs):
            self.calls += 1
            return super().forward(inputs)

    model = CountingLinear(2, 2).double()
    inputs = torch.randn(10, 2, dtype=torch.float64)
    targets = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=5)
    gradients = list(
        _candidate_gradients(
            model,
            nn.CrossEntropyLoss(),
            loader,
            list(model.parameters()),
            torch.device("cpu"),
        )
    )
    assert len(gradients) == 10
    assert model.calls == 2
