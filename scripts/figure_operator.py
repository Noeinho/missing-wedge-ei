from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor

from missing_wedge_ei.data import DATA_ROOT
from missing_wedge_ei.forward_op import measure, wedge_mask

ROOT = Path(__file__).resolve().parent.parent
N = 32
HALF_ANGLE = 30.0
N_IMAGES = 3


def main():
    dataset = torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=ToTensor()
    )
    x = torch.stack([dataset[i][0] for i in range(N_IMAGES)])  # (3, 1, 28, 28)
    x = F.pad(x, (2, 2, 2, 2))                                 # (3, 1, 32, 32)

    mask = wedge_mask(N, HALF_ANGLE)
    y = measure(x, mask)
    d = x - y # part removed by A

    print("fraction of frequencies kept:", mask.mean().item())
    print("mean of x:", x.mean(dim=(1, 2, 3)))
    print("mean of y:", y.mean(dim=(1, 2, 3)))
    print("min of x, min of y:", x.min().item(), y.min().item())

    mask_display = torch.fft.fftshift(mask)  # shift for display only (zero frequency at the centre)
    vmin = min(x.min().item(), y.min().item())
    vmax = max(x.max().item(), y.max().item())
    dmax = d.abs().max().item()

    fig, axes = plt.subplots(N_IMAGES, 4, figsize=(8, 2 * N_IMAGES))
    for i in range(N_IMAGES):
        axes[i, 0].imshow(x[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 1].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(y[i, 0], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 3].imshow(d[i, 0], cmap="RdBu", vmin=-dmax, vmax=dmax)
        for ax in axes[i]:
            ax.axis("off")
    for ax, title in zip(axes[0], ["x", "mask (fftshift)", "y = A x", "x - y"]):
        ax.set_title(title)

    out = ROOT / "figures" / "operator.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure saved to", out)
    plt.show()


if __name__ == "__main__":
    main()