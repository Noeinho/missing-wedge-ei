import torch
import math
from torch.fft import fft2, ifft2

def wedge_mask(n: int, half_angle_deg:float) -> torch.Tensor:
    """Fourier mask of the missing wedge: 1 where the frequency is measured, 0 where it is missing.
    n : image size (n x n)
    half_angle_deg : half-angle of the cone in degrees (total angle = 2 * half_angle_deg)
    """
    k = torch.fft.fftfreq(n)  # the n frequencies k/n, in FFT order (zero first)
    kx, ky = torch.meshgrid(k, k, indexing="xy")  # frequency grid (kx, ky) of shape (n, n)

    # The mask is 1 outside the cone, i.e. where |kx| > |ky| * tan(half_angle).
    # The cone is around the ky axis. kx and ky are the true frequencies up to a
    # common sampling factor, which does not change the comparison.
    mask = torch.abs(kx) > torch.abs(ky) * math.tan(math.radians(half_angle_deg))

    mask = mask.to(torch.float32)
    mask[0, 0] = 1.0   # keep the zero frequency (mean of the image)
    return mask

def measure(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the operator A, i.e. remove the masked frequencies.
    x: (B, C, N, N) : batch of images
    mask: (N, N) : Fourier mask of the missing wedge"""
    return ifft2(mask * fft2(x)).real  # equivalent to a convolution in image space

if __name__ == "__main__":
    print(wedge_mask(4, 30.))      # only when forward_op.py is run directly, not when imported