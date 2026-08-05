import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from flightrec.probes.curvature import hutchinson_trace, lanczos_spectrum
from flightrec.probes.hvp import HessianOperator


def test_spectrum_and_trace_match_exact_hessian():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    inputs = torch.randn(12, 3, dtype=torch.float64)
    targets = torch.randint(0, 2, (12,))
    flat = torch.cat([param.detach().flatten() for param in model.parameters()])

    def functional_loss(vector):
        weight1 = vector[:12].reshape(4, 3)
        bias1 = vector[12:16]
        weight2 = vector[16:24].reshape(2, 4)
        bias2 = vector[24:26]
        hidden = torch.tanh(functional.linear(inputs, weight1, bias1))
        return functional.cross_entropy(functional.linear(hidden, weight2, bias2), targets)

    exact = torch.autograd.functional.hessian(functional_loss, flat).numpy()
    operator = HessianOperator(
        model, lambda: functional.cross_entropy(model(inputs), targets), "cpu"
    )
    low, high = lanczos_spectrum(operator, k=2)
    eigenvalues = np.linalg.eigvalsh(exact)
    np.testing.assert_allclose(low, eigenvalues[:2], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(high, eigenvalues[-2:], rtol=1e-4, atol=1e-7)
    assert eigenvalues[0] < -1e-4
    estimate = hutchinson_trace(operator, n_probes=256, seed=0)
    assert abs(estimate - np.trace(exact)) / abs(np.trace(exact)) < 0.15
