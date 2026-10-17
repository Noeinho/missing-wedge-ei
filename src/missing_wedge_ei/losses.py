import torch
import torch.nn.functional as F

from missing_wedge_ei.forward_op import measure


def consistency_loss(model, x_b, mask):
    y = measure(x_b, mask)
    return F.mse_loss(measure(model(y), mask), y)


def make_equivariant_loss(alpha=1.0, detach=True):
    """returns the loss function with the right parameters. needed because we need extra parameters alpha and detach here."""

    def equivariant_loss(model, x_b, mask):
        y1 = measure(x_b, mask)
        x1 = model(y1)

        x1_rot = (
            x1.detach() if detach else x1
        )  # puisque x1 va être utilisé comme étiquette par la suite, il doit etre vu comme cst à partir de maintenant.

        k = (
            1 if torch.rand(()) < 0.5 else 3
        )  # on veut k=1 ou 3 equiproba pour tourner l'image et changer la zone manquante.
        x2 = torch.rot90(x1_rot, k, dims=(2, 3))
        y2 = measure(x2, mask)

        return F.mse_loss(measure(x1, mask), y1) + alpha * F.mse_loss(model(y2), x2)

    return equivariant_loss  # il faut renvoyer une fonction. loss_fn = make_equivariant_loss(alpha, detach).
