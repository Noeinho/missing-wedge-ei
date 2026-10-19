import torch
import torch.nn.functional as F

from missing_wedge_ei.forward_op import measure


def consistency_loss(model, x_b, mask):
    y = measure(x_b, mask)
    return F.mse_loss(measure(model(y), mask), y)


def make_equivariant_loss(alpha=1.0, detach=True):
    """Return the equivariant loss function with alpha and detach fixed."""

    def equivariant_loss(model, x_b, mask):
        y1 = measure(x_b, mask)
        x1 = model(y1)

        # x1 is used as a target below, so it is treated as a constant when detach=True
        x1_rot = x1.detach() if detach else x1

        # k = 1 or 3 with equal probability: the rotation moves the missing cone
        k = 1 if torch.rand(()) < 0.5 else 3
        x2 = torch.rot90(x1_rot, k, dims=(2, 3))
        y2 = measure(x2, mask)

        return F.mse_loss(measure(x1, mask), y1) + alpha * F.mse_loss(model(y2), x2)

    return equivariant_loss
