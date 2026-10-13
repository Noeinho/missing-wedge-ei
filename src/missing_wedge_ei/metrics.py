import torch

def psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """Calcule le PSNR entre deux images x_hat et x. 
    On suppose que les images sont dans [0, 1]."""
    mse = torch.mean((x_hat - x) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1 / mse).item()