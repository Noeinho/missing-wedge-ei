"""Premier entraînement supervisé : entraîné sur le train, évalué sur la validation."""

import argparse
import json
import time
from pathlib import Path

import torch

from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.forward_op import measure, wedge_mask
from missing_wedge_ei.metrics import psnr
from missing_wedge_ei.model import ConvNet
from missing_wedge_ei.train import supervised_loss, train

ROOT = Path(__file__).resolve().parent.parent
DATA_SEED = 0  # rotations des données : fixées une fois pour toutes
HALF_ANGLE = 30.0
N_TRAIN = 55_000


def train_supervised(x_train, x_val, mask, residual, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)  # même initialisation à chaque appel
    model = ConvNet(residual=residual)
    baseline = psnr(measure(x_val, mask), x_val)  # PSNR avant tout entraînement
    start = time.perf_counter()
    history = train(
        model,
        supervised_loss,
        x_train,
        x_val,
        mask,
        epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )
    seconds = time.perf_counter() - start
    return model, baseline, history, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants", nargs="+", choices=["residual", "direct"], default=["residual"]
    )
    args = parser.parse_args()

    x_all = load_mnist32(train=True, seed=DATA_SEED)
    x_train, x_val = x_all[:N_TRAIN], x_all[N_TRAIN:]
    mask = wedge_mask(32, HALF_ANGLE)

    for variant in args.variants:
        residual = variant == "residual"
        model, baseline, hist, seconds = train_supervised(
            x_train,
            x_val,
            mask,
            residual,
            args.epochs,
            args.lr,
            args.batch_size,
            args.seed,
        )

        losses, psnrs = hist["train_loss"], hist["val_psnr"]
        per_epoch = seconds / args.epochs
        print(f"\n{variant} (seed {args.seed})")
        print(f"  PSNR de la baseline sur la validation : {baseline:.2f} dB")
        print(f"  perte d'entraînement : {losses[0]:.5f} -> {losses[-1]:.5f}")
        print(
            f"  PSNR de validation   : {psnrs[0]:.2f} -> {psnrs[-1]:.2f} dB "
            f"({psnrs[-1] - baseline:+.2f} dB par rapport à la baseline)"
        )
        print(
            f"  temps : {seconds:.0f} s au total, {per_epoch:.0f} s par époque "
            f"(validation comprise)"
        )

        result = {
            "method": "supervised",
            "variant": variant,
            "residual": residual,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "data_seed": DATA_SEED,
            "half_angle_deg": HALF_ANGLE,
            "n_train": N_TRAIN,
            "n_val": int(x_val.shape[0]),
            "baseline_val_psnr_db": float(baseline),
            "history": hist,
            "seconds_total": seconds,
            "seconds_per_epoch": per_epoch,
            "torch_version": torch.__version__,
        }
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        name = f"supervised_{variant}_seed{args.seed}"
        (results_dir / f"{name}.json").write_text(json.dumps(result, indent=2))

        runs_dir = ROOT / "runs"  # ignoré par Git
        runs_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), runs_dir / f"{name}.pt")


if __name__ == "__main__":
    main()
