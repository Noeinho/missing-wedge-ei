import torch
from missing_wedge_ei.forward_op import wedge_mask, measure

torch.manual_seed(0)

N = 32
MASK = wedge_mask(N, 30.0)
X = torch.rand(2,1,N,N) #batch de 2 images 1 canal, taille N x N

def test_measure_keeps_shape():
    """Test que l'opérateur A garde la même forme que l'image d'entrée."""
    y = measure(X, MASK)
    assert y.shape == X.shape, f"Expected shape {X.shape}, but got {y.shape}"

def test_measure_is_projection():
    """Test que l'opérateur A est une projection, ie A(A(x)) = A(x)"""
    y = measure(X, MASK)
    assert torch.allclose(measure(y, MASK), y, atol=1e-6), "measure is not a projection"

def test_mask_4x4():
    expected = torch.tensor([[1., 1., 1., 1.],
                             [0., 1., 1., 1.],
                             [0., 0., 1., 0.],
                             [0., 1., 1., 1.]])
    assert torch.equal(wedge_mask(4, 30.), expected) #== renverrait un tenseur avec 16 fois True

def test_mask_sum():
    assert wedge_mask(32, 30.).sum() == 727.

def test_mean_conserved():
    """Test que l'opérateur A conserve la moyenne de l'image."""
    y = measure(X, MASK)
    assert torch.allclose(y.mean(), X.mean(), atol=1e-6), "measure does not conserve mean"