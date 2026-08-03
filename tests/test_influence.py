import copy

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_blobs
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flightrec.analysis.influence import InfluenceConfig, influence_on


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
