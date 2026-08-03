import torch
from torch import nn

from flightrec.probes.gradsanity import check_gradients, check_model_graph


class CorrectSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        return 2 * value * grad_output


class WrongSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        return value * grad_output


def test_correct_custom_function_passes_complex_step():
    params = {"x": torch.tensor([0.3, -0.7])}
    report = check_gradients(lambda values: CorrectSquare.apply(values["x"]).sum(), params)
    assert report.passed
    assert report.max_abs_err_cs is not None and report.max_abs_err_cs < 1e-12


def test_wrong_backward_is_flagged_by_both_checks():
    params = {"x": torch.tensor([0.3, -0.7])}
    report = check_gradients(lambda values: WrongSquare.apply(values["x"]).sum(), params)
    assert not report.passed
    assert report.max_abs_err_fd > 0.1
    assert report.max_abs_err_cs is not None and report.max_abs_err_cs > 0.1


def test_detached_model_branch_is_unreachable():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.connected = nn.Linear(2, 1)
            self.detached = nn.Linear(2, 1)

        def forward(self, inputs):
            return self.connected(inputs) + self.detached(inputs).detach()

    model = Broken()
    report = check_model_graph(model, model(torch.ones(2, 2)).sum())
    assert set(report.unreachable_params) == {"detached.weight", "detached.bias"}
