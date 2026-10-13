import torch
from missing_wedge_ei.forward_op import measure

@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> float:
    """PSNR moyen (en dB) sur le batch, images supposées dans [0, 1].
    x_hat: (B, C, N, N) : batch d'images reconstruites
    x: (B, C, N, N) : batch d'images originales"""
    mse = torch.mean((x_hat - x) ** 2, dim=(1, 2, 3))      # (B,)
    return torch.mean(10 * torch.log10(1 / mse.clamp_min(eps))).item() #clamp_min fait que les valeurs sont au moins eps.

def split_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Renvoie la MSE orthogonale et parallèle à l'opérateur A, ie freq perdues et conservées.
    x_hat: (B, C, N, N) : batch d'images reconstruites
    x: (B, C, N, N) : batch d'images originales
    mask: (N, N) : masque de Fourier du missing wedge"""
    projection = measure(x_hat - x, mask)
    mse_parallel = torch.mean(projection ** 2)
    mse_orthogonal = torch.mean((x_hat - x - projection) ** 2)
    return mse_parallel.item(), mse_orthogonal.item()