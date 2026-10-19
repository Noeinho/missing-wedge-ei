import torch
from missing_wedge_ei.forward_op import wedge_mask, measure

torch.manual_seed(0)

N = 32
MASK = wedge_mask(N, 30.0)
X = torch.rand(2,1,N,N) # batch of 2 single-channel images of size N x N

def test_measure_keeps_shape():
    """The operator A keeps the shape of the input image."""
    y = measure(X, MASK)
    assert y.shape == X.shape, f"Expected shape {X.shape}, but got {y.shape}"

def test_measure_is_projection():
    """The operator A is a projection: A(A(x)) = A(x)."""
    y = measure(X, MASK)
    assert torch.allclose(measure(y, MASK), y, atol=1e-6), "measure is not a projection"

def test_mask_4x4():
    expected = torch.tensor([[1., 1., 1., 1.],
                             [0., 1., 1., 1.],
                             [0., 0., 1., 0.],
                             [0., 1., 1., 1.]])
    assert torch.equal(wedge_mask(4, 30.), expected) # == would return a tensor of 16 booleans

def test_mask_sum():
    assert wedge_mask(32, 30.).sum() == 727.

def test_mean_conserved():
    """The operator A preserves the mean of the image."""
    y = measure(X, MASK)
    assert torch.allclose(y.mean(), X.mean(), atol=1e-6), "measure does not conserve mean"