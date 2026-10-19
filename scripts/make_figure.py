"""README figure: truth, measurement, equivariant reconstruction, error.

Uses the weights saved by run_experiments.py (runs/ folder, ignored by Git).
Writes figures/reconstructions.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.forward_op import measure, wedge_mask
from missing_wedge_ei.metrics import psnr
from missing_wedge_ei.model import ConvNet

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "equivariant_residual_a3_detach_e20_lr0.003_final_seed0.pt"
N_IMAGES = 6
HALF_ANGLE = 30.0


def main():
    x = load_mnist32(train=False, seed=1)[:N_IMAGES]   # test images
    mask = wedge_mask(32, HALF_ANGLE)
    y = measure(x, mask)

    model = ConvNet(residual=True)
    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()
    with torch.no_grad():
        x_hat = model(y)

    print("PSNR on these images, measurement:", round(psnr(y, x), 2), "dB")
    print("PSNR on these images, reconstruction:", round(psnr(x_hat, x), 2), "dB")

    error = x - x_hat
    vmin = min(x.min().item(), y.min().item(), x_hat.min().item())
    vmax = max(x.max().item(), y.max().item(), x_hat.max().item())
    emax = error.abs().max().item()

    rows = [
        ("x (truth)", x, "gray", vmin, vmax),
        ("y = A x", y, "gray", vmin, vmax),
        ("reconstruction", x_hat, "gray", vmin, vmax),
        ("x - reconstruction", error, "RdBu", -emax, emax),
    ]
    fig, axes = plt.subplots(len(rows), N_IMAGES, figsize=(1.6 * N_IMAGES, 6.6))
    for r, (label, data, cmap, lo, hi) in enumerate(rows):
        for c in range(N_IMAGES):
            ax = axes[r, c]
            ax.imshow(data[c, 0], cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=8)

    out = ROOT / "figures" / "reconstructions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure saved to", out)


if __name__ == "__main__":
    main()
