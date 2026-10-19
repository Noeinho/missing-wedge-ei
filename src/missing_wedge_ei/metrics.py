import torch
from missing_wedge_ei.forward_op import measure

@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean PSNR (in dB) over the batch, images assumed to be in [0, 1].
    x_hat: (B, C, N, N) : batch of reconstructed images
    x: (B, C, N, N) : batch of original images"""
    mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))      # (B,)
    return torch.mean(10 * torch.log10(1 / mse.clamp_min(eps))).item()   # clamp_min keeps the MSE at least eps

def split_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Return the MSE parallel and orthogonal to the operator A, i.e. on the
    measured and on the missing frequencies (in this order). They sum to the total MSE.
    x_hat: (B, C, N, N) : batch of reconstructed images
    x: (B, C, N, N) : batch of original images
    mask: (N, N) : Fourier mask of the missing wedge"""
    projection = measure(x_hat - x, mask)
    mse_parallel = torch.mean(projection ** 2)
    mse_orthogonal = torch.mean((x_hat - x - projection) ** 2)
    return mse_parallel.item(), mse_orthogonal.item()