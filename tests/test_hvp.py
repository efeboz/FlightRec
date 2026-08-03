import torch
from torch import nn

from flightrec.probes.hvp import hvp


def test_hvp_matches_exact_hessian():
    torch.manual_seed(0)
    model = nn.Linear(3, 2, bias=True).double()
    inputs = torch.randn(5, 3, dtype=torch.float64)
    targets = torch.randn(5, 2, dtype=torch.float64)
    params = list(model.parameters())
    flat = torch.cat([param.detach().flatten() for param in params])

    def functional_loss(vector):
        weight = vector[:6].reshape(2, 3)
        bias = vector[6:]
        return ((inputs @ weight.T + bias - targets) ** 2).mean()

    exact = torch.autograd.functional.hessian(functional_loss, flat)
    vector = torch.randn_like(flat)
    actual = hvp(lambda: ((model(inputs) - targets) ** 2).mean(), params, vector)
    torch.testing.assert_close(actual, exact @ vector, rtol=1e-5, atol=1e-7)
