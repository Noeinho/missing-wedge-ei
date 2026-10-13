import torch

from missing_wedge_ei.metrics import psnr

N = 32
X = torch.rand(1, N, N)  # image 1 canal, taille N x N
X_hat = X + 0.1

def test_psnr():
    epsilon = 1e-5
    assert abs(psnr(X_hat, X) - 20.) < epsilon