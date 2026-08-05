import torch
from torch import nn
from torch.nn import functional as functional

from flightrec.probes.hvp import hvp


def test_hvp_matches_exact_hessian():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    inputs = torch.randn(7, 3, dtype=torch.float64)
    targets = torch.randint(0, 2, (7,))
    params = list(model.parameters())
    flat = torch.cat([param.detach().flatten() for param in params])

    def functional_loss(vector):
        weight1 = vector[:12].reshape(4, 3)
        bias1 = vector[12:16]
        weight2 = vector[16:24].reshape(2, 4)
        bias2 = vector[24:26]
        hidden = torch.tanh(functional.linear(inputs, weight1, bias1))
        return functional.cross_entropy(functional.linear(hidden, weight2, bias2), targets)

    exact = torch.autograd.functional.hessian(functional_loss, flat)
    vector = torch.randn_like(flat)
    actual = hvp(lambda: functional.cross_entropy(model(inputs), targets), params, vector)
    torch.testing.assert_close(actual, exact @ vector, rtol=1e-5, atol=1e-7)
    assert bool(torch.linalg.eigvalsh(exact).min() < -1e-4)


def test_hvp_uses_double_backward_safe_attention_backend_on_cpu():
    torch.manual_seed(3)
    query = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 2, 4, dtype=torch.float64)
    value = torch.randn(1, 2, 4, dtype=torch.float64)
    vector = torch.randn(query.numel(), dtype=torch.float64)

    def loss():
        output = functional.scaled_dot_product_attention(query, key, value)
        return output.square().mean()

    result = hvp(loss, [query], vector)
    assert result.shape == vector.shape
    assert bool(torch.isfinite(result).all())
