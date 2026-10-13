import torch
import pytest
from missing_wedge_ei.metrics import psnr, split_mse
from missing_wedge_ei.forward_op import wedge_mask, measure

torch.manual_seed(0)
N = 32
MASK = wedge_mask(N, 30.0)

def test_psnr():
    X = torch.rand(1, 1, N, N)  # batch d'1 image 1 canal, taille N x N
    assert psnr(X + 0.1, X) == pytest.approx(20.0, abs=1e-3)

def test_average_psnr():
    X = torch.rand(1, 1, N, N)
    X_shift = X + 0.01
    X_hat = X + 0.1
    X_hat_batch = torch.cat([X_hat, X_shift])  # batch de 2 images
    X_batch = torch.cat([X, X])                 # batch de 2 images
    assert psnr(X_hat_batch, X_batch) == pytest.approx(30.0, abs=1e-3)

def test_type_psnr():
    X = torch.rand(1, 1, N, N)
    assert isinstance(psnr(X + 0.1, X), float)

def test_psnr_perfect_reconstruction_is_finite():
    x = torch.rand(2, 1, N, N)
    assert psnr(x, x) == pytest.approx(120.0, abs=1e-3)   # plafond imposé par eps=1e-12

def test_split_mse():
    x = torch.rand(4, 1, N, N)
    x_hat = x + 0.1 * torch.randn(4, 1, N, N)
    measured, missing = split_mse(x_hat, x, MASK) #ie parallel and orthogonal
    total = torch.mean((x_hat - x) ** 2).item()
    assert measured + missing == pytest.approx(total, rel=1e-4) #mses should add up with Pythagore.

def test_split_mse_baseline_error_is_all_in_the_cone():
    x = torch.rand(4, 1, N, N)
    x_hat = measure(x, MASK)             # reconstruction naïve : x_hat = y
    measured, missing = split_mse(x_hat, x, MASK)
    assert missing > 0
    assert measured < 1e-6 * missing


def test_split_mse_error_in_measured_part_only():
    x = torch.rand(4, 1, N, N)
    x_hat = x + measure(torch.randn(4, 1, N, N), MASK)   # erreur = A(bruit)
    measured, missing = split_mse(x_hat, x, MASK)
    assert measured > 0
    assert missing < 1e-6 * measured