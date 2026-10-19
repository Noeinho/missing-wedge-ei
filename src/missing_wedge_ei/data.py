from pathlib import Path

import torch
import torchvision
from torchvision.transforms import ToTensor
import torch.nn.functional as F


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"  # repository root, not the cwd


def load_mnist32(train: bool, seed: int = 0, rotate: bool = True,
                 root: str = str(DATA_ROOT)) -> torch.Tensor:
    """MNIST as (N, 1, 32, 32), values in [0, 1].

    If rotate=True, each image gets k quarter turns, k drawn uniformly in
    {0, 1, 2, 3}: the dataset distribution is then exactly invariant under
    90-degree rotation.
    """
    dataset = torchvision.datasets.MNIST(
        root=root, train=train, download=True, transform=ToTensor()
    )
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])  # (N, 1, 28, 28)
    x = F.pad(x, (2, 2, 2, 2))                                     # (N, 1, 32, 32)

    if rotate:
        g = torch.Generator().manual_seed(seed)
        k = torch.randint(0, 4, (x.shape[0],), generator=g)        # (N,)
        out = x.clone()
        for r in [1, 2, 3]:
            idx = k == r                                           # boolean mask (N,)
            out[idx] = torch.rot90(x[idx], r, dims=(2, 3))
        x = out
    return x


if __name__ == "__main__":
    x = load_mnist32(train=True, seed=0, rotate=True)
    print(x.shape)
    # display a grid of 10 images
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()