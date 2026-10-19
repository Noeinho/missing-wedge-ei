"""Sanity check: can the model and the training loop memorise 64 images?
If the loss does not decrease here, there is a bug.
"""

import torch

from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.forward_op import measure, wedge_mask
from missing_wedge_ei.metrics import psnr
from missing_wedge_ei.model import ConvNet
from missing_wedge_ei.train import supervised_loss, train


def overfit_batch(x, mask, residual, epochs=300, lr=1e-3, seed=0):
    baseline = psnr(measure(x, mask), x)  # PSNR before any training
    torch.manual_seed(seed)
    model = ConvNet(residual=residual)
    # batch_size = number of images: one epoch is a single step on the same batch.
    history = train(
        model,
        supervised_loss,
        x,
        x,
        mask,
        epochs,
        lr=lr,
        batch_size=x.shape[0],
        seed=seed,
    )
    return baseline, history


if __name__ == "__main__":
    x = load_mnist32(train=True, seed=0)[:64]
    mask = wedge_mask(32, 30.0)

    for residual in (True, False):
        baseline, hist = overfit_batch(x, mask, residual)
        losses, psnrs = hist["train_loss"], hist["val_psnr"]
        print(f"\nresidual={residual}")
        print(f"  baseline PSNR on these images : {baseline:.2f} dB")
        print(f"  loss: {losses[0]:.5f} -> {losses[-1]:.5f}")
        print(
            f"  PSNR: {psnrs[0]:.2f} -> {psnrs[-1]:.2f} dB (on the training images)"
        )
