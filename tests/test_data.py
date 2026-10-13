import torch
from missing_wedge_ei.data import load_mnist32

x = load_mnist32(train=False, seed=0, rotate=True)

def test_form_and_type():
    """Test que le dataset est bien de la forme (N, 1, 32, 32) et de type float32."""
    assert x.ndim == 4, f"Expected 4 dimensions, but got {x.ndim}"
    assert x.shape[1] == 1, f"Expected 1 channel, but got {x.shape[1]}"
    assert x.shape[2] == 32 and x.shape[3] == 32, f"Expected shape (N, 1, 32, 32), but got {x.shape}"
    assert x.dtype == torch.float32, f"Expected dtype float32, but got {x.dtype}"

def test_values_range():
    """Test que les valeurs sont bien dans [0, 1]."""
    max = x.max().item()
    assert x.min() >= 0 and max <= 1 and max > 0.99, f"Expected values in [0, 1], but got min {x.min()} and max {x.max()}"

def test_reproducibility():
    """Test que le dataset est reproductible avec le même seed."""
    x1 = load_mnist32(train=False, seed=0, rotate=True)
    x2 = load_mnist32(train=False, seed=1, rotate=True)
    assert torch.equal(x1, x), "Datasets with the same seed are not equal"
    assert torch.equal(x1, x2) == False, "Datasets with different seeds are equal"
