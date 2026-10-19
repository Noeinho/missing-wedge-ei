import torch
from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.metrics import psnr
import torch.nn.functional as F
import torch.optim as optim

from missing_wedge_ei.forward_op import measure


def supervised_loss(model, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE between the reconstruction f(A x) and the true image x.
    x : (B, 1, 32, 32)"""
    y = measure(x, mask)
    # the forward pass is inside the loss function: the equivariant loss calls the model several times
    return F.mse_loss(model(y), x)


@torch.no_grad()
def evaluate(
    model, x: torch.Tensor, mask: torch.Tensor, batch_size: int = 500
) -> float:
    """Mean PSNR over the evaluation set.
    x : (N, 1, 32, 32), the whole evaluation set"""

    was_training = model.training
    model.eval()

    mean_psnr = 0.0
    n = x.shape[0]
    for i in range(0, n, batch_size):
        x_batch = x[i : i + batch_size]  # the last batch may be smaller
        mean_psnr += psnr(model(measure(x_batch, mask)), x_batch) * x_batch.shape[0]

    model.train(was_training)
    return mean_psnr / n


def train(
    model,
    loss_fn,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    mask: torch.Tensor,
    epochs,
    lr=1e-3,
    batch_size=128,
    seed=0,
):
    """Train the model with the given loss function (Adam, shuffled mini-batches)."""

    g = torch.Generator().manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = x_train.shape[0]

    history = {"train_loss": [], "val_psnr": []}

    for i in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total_loss = 0.0

        for j in range(0, n, batch_size):
            optimizer.zero_grad()
            x_batch = x_train[perm[j : j + batch_size]]

            loss = loss_fn(model, x_batch, mask)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * x_batch.shape[0]

        val_psnr = evaluate(model, x_val, mask)
        history["train_loss"].append(total_loss / n)
        history["val_psnr"].append(val_psnr)
        print(
            f"Epoch {i + 1}/{epochs} | "
            f"train loss {total_loss / n:.5f} | val PSNR {val_psnr:.2f} dB"
        )

    return history
