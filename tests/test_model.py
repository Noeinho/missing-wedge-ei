import pytest

import torch
from missing_wedge_ei.model import ConvNet

SEED = 1
N = 32


@pytest.fixture(autouse=True)  # re-seed before every test
def seed_everything():
    torch.manual_seed(SEED)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    "residual", [True, False]
)  # run the test for each listed value
def test_ConvNet_shape(batch_size, residual):
    X_batch = torch.rand(batch_size, 1, N, N)
    conv_model = ConvNet(residual=residual)
    assert conv_model(X_batch).shape == X_batch.shape


def test_ConvNet_residual_is_identity():
    conv_model = ConvNet()
    X = torch.rand(1, 1, N, N)
    assert torch.equal(
        conv_model(X), X
    )  # calling the module runs forward


def test_ConvNet_no_residual_isnot_identity():
    conv_model = ConvNet(residual=False)
    X = torch.rand(1, 1, N, N)
    assert not torch.equal(
        conv_model(X), X
    )  # calling the module runs forward


@pytest.mark.parametrize("residual", [True, False])
def test_ConvNet_nb_parameter(residual):
    conv_model = ConvNet(residual=residual)
    assert sum(p.numel() for p in conv_model.parameters()) == 28353


@pytest.mark.parametrize("residual", [True, False])
def test_ConvNet_all_grads_active(residual):
    conv_model = ConvNet(residual=residual)
    X = torch.rand(2, 1, N, N)
    conv_model(X).sum().backward()
    for name, p in conv_model.named_parameters():
        assert p.grad is not None, name
