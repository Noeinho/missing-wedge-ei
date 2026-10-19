import copy

import pytest
import torch

from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.forward_op import measure, wedge_mask
from missing_wedge_ei.metrics import psnr
from missing_wedge_ei.model import ConvNet
from missing_wedge_ei.train import evaluate, supervised_loss, train

SEED = 1
N = 32
MASK = wedge_mask(N, 30.0)


@pytest.fixture(autouse=True)
def seed_everything():
    """Same seed before each test, whatever the execution order."""
    torch.manual_seed(SEED)


@pytest.fixture(scope="module")
def data():
    """256 real MNIST images (test split, with rotations), loaded once.

    Structured images rather than noise: noise has nothing to learn, so the loss
    might not decrease even with correct code.
    """
    return load_mnist32(train=False, seed=1)[:256]


def _run(model, data, epochs, loss_fn=supervised_loss, n_train=128, **kwargs):
    """Train on data[:n_train], validate on the next 64 images."""
    x_train = data[:n_train]
    x_val = data[n_train : n_train + 64]
    return train(model, loss_fn, x_train, x_val, MASK, epochs, **kwargs)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_of_zero_init_model_equals_baseline(data):
    # 200 images with batch_size=64: batches of 64, 64, 64 and 8 (tests the last batch).
    x = data[:200]
    expected = psnr(measure(x, MASK), x)
    got = evaluate(ConvNet(residual=True), x, MASK, batch_size=64)
    assert got == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("was_training", [True, False])
def test_evaluate_restores_the_mode_of_the_model(data, was_training):
    model = ConvNet(residual=True)
    model.train(was_training)
    evaluate(model, data[:64], MASK, batch_size=32)
    assert model.training == was_training


# --------------------------------------------------------------------------- #
# train: structure
# --------------------------------------------------------------------------- #


def test_history_has_one_entry_per_epoch(data):
    history = _run(ConvNet(residual=True), data, epochs=3, batch_size=64)
    assert set(history) == {"train_loss", "val_psnr"}
    assert len(history["train_loss"]) == 3
    assert len(history["val_psnr"]) == 3


def test_loss_function_receives_batches_not_the_whole_set(data):
    sizes = []

    def spy(model, xb, mask):
        sizes.append(xb.shape[0])
        return supervised_loss(model, xb, mask)

    # 100 = 3 * 32 + 4: the last batch is smaller.
    _run(
        ConvNet(residual=True), data, epochs=1, loss_fn=spy, n_train=100, batch_size=32
    )
    assert sizes == [32, 32, 32, 4]


def test_every_training_image_is_seen_exactly_once_per_epoch(data):
    batches = []

    def spy(model, xb, mask):
        batches.append(xb.clone())
        return supervised_loss(model, xb, mask)

    _run(
        ConvNet(residual=True), data, epochs=1, loss_fn=spy, n_train=100, batch_size=32
    )
    seen = torch.cat(batches)
    assert seen.shape[0] == 100
    # Same multiset of images (compared through sorted pixel sums).
    assert torch.allclose(
        seen.sum(dim=(1, 2, 3)).sort().values,
        data[:100].sum(dim=(1, 2, 3)).sort().values,
    )


# --------------------------------------------------------------------------- #
# train: optimisation works
# --------------------------------------------------------------------------- #


def test_training_loss_decreases(data):
    history = _run(ConvNet(residual=True), data, epochs=15, n_train=192, batch_size=32)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_parameters_change_during_training(data):
    model = ConvNet(residual=True)
    before = copy.deepcopy(model.state_dict())
    _run(model, data, epochs=1, batch_size=32)
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_zero_learning_rate_leaves_parameters_unchanged(data):
    # Catches an ignored lr (hard-coded in the optimizer).
    model = ConvNet(residual=False)  # non-zero weights
    before = copy.deepcopy(model.state_dict())
    _run(model, data, epochs=1, lr=0.0, batch_size=32)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), k


# --------------------------------------------------------------------------- #
# train: reproducibility and seed
# --------------------------------------------------------------------------- #


def _history_for(data, shuffle_seed):
    torch.manual_seed(SEED)  # same model initialisation
    model = ConvNet(residual=True)
    return _run(model, data, epochs=2, batch_size=32, seed=shuffle_seed)


def test_same_seed_gives_same_history(data):
    h1 = _history_for(data, shuffle_seed=0)
    h2 = _history_for(data, shuffle_seed=0)
    assert h1["train_loss"] == pytest.approx(h2["train_loss"], rel=1e-6)
    assert h1["val_psnr"] == pytest.approx(h2["val_psnr"], rel=1e-6)


def test_shuffle_seed_is_actually_used(data):
    # Catches an ignored seed argument: two different shuffles, two trajectories.
    h1 = _history_for(data, shuffle_seed=0)
    h2 = _history_for(data, shuffle_seed=1)
    assert h1["train_loss"] != h2["train_loss"]
