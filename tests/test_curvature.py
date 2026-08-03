import numpy as np
import torch
from torch import nn

from flightrec.probes.curvature import hutchinson_trace, lanczos_spectrum
from flightrec.probes.hvp import HessianOperator


def test_spectrum_and_trace_match_exact_hessian():
    torch.manual_seed(1)
    model = nn.Linear(3, 2).double()
    inputs = torch.randn(12, 3, dtype=torch.float64)
    targets = torch.randn(12, 2, dtype=torch.float64)
    flat = torch.cat([param.detach().flatten() for param in model.parameters()])

    def functional_loss(vector):
        return ((inputs @ vector[:6].reshape(2, 3).T + vector[6:] - targets) ** 2).mean()

    exact = torch.autograd.functional.hessian(functional_loss, flat).numpy()
    operator = HessianOperator(model, lambda: ((model(inputs) - targets) ** 2).mean(), "cpu")
    low, high = lanczos_spectrum(operator, k=2)
    eigenvalues = np.linalg.eigvalsh(exact)
    np.testing.assert_allclose(low, eigenvalues[:2], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(high, eigenvalues[-2:], rtol=1e-4, atol=1e-7)
    estimate = hutchinson_trace(operator, n_probes=256, seed=0)
    assert abs(estimate - np.trace(exact)) / np.trace(exact) < 0.15
