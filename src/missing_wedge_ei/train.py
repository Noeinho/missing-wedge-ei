import torch
from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.metrics import psnr
import torch.nn.functional as F
import torch.optim as optim

from missing_wedge_ei.forward_op import measure


def supervised_loss(model, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE entre la reconstruction f(A x) et la vraie image x.
    Il est important que la loss renvoie un tenseur et pas un float, pour qu'il ait une méthode backward.
    x : (B, 1, 32, 32)"""
    y = measure(x, mask)
    return F.mse_loss(
        model(y), x
    )  # attention on met la forward dans la fn de perte. C'est un choix qui se révelera utile en équivariant.


@torch.no_grad()  # équivalent à commencer la function par with torch.no_grad()
def evaluate(
    model, x: torch.Tensor, mask: torch.Tensor, batch_size: int = 500
) -> float:
    """Renvoie le PSNR moyen sur le jeu d'évaluation.
    x : (N, 1, 32, 32) = x_eval complet"""

    was_training = model.training
    model.eval()  # habitude à prendre. hérité de nn.Module.

    mean_psnr = 0.0
    n = x.shape[0]
    for i in range(0, n, batch_size):  # on fait des pas de longueur batch size
        x_batch = x[
            i : i + batch_size
        ]  # si i+batch_size dépasse la fin du tensor, torch gère ça et s'arrête là ou il faut.
        mean_psnr += psnr(model(measure(x_batch, mask)), x_batch) * x_batch.shape[0]

    model.train(was_training)  # habitude à prendre.
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
    """Entraîne le modèle en utilisant la fonction de perte spécifiée."""

    g = torch.Generator().manual_seed(seed)

    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )  # parametres retrouvés automatiquement car usage de nn.Module
    n = x_train.shape[0]

    history = {"train_loss": [], "val_psnr": []}

    for i in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total_loss = 0.0

        for j in range(0, n, batch_size):
            optimizer.zero_grad()
            x_batch = x_train[
                perm[j : j + batch_size]
            ]  # pour faire des mini-batchs stochastiques.

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
