"""Lance une série d'entraînements et enregistre un JSON par run.

Pour changer les expériences, modifier la liste RUNS en bas du fichier.
Chaque run écrit results/<nom>.json (versionné) et runs/<nom>.pt (ignoré par Git).
Relancer une même configuration avec la même graine écrase ses fichiers.

final_test=True évalue aussi le jeu de test : à ne faire qu'une seule fois par
configuration finale, après les réglages sur la validation.
"""

import json
import time
from pathlib import Path

import torch

from missing_wedge_ei.data import load_mnist32
from missing_wedge_ei.forward_op import measure, wedge_mask
from missing_wedge_ei.losses import consistency_loss, make_equivariant_loss
from missing_wedge_ei.metrics import psnr, split_mse
from missing_wedge_ei.model import ConvNet
from missing_wedge_ei.train import evaluate, supervised_loss, train

ROOT = Path(__file__).resolve().parent.parent
DATA_SEED = 0  # rotations du jeu d'entraînement et de validation
TEST_SEED = 1  # rotations du jeu de test
HALF_ANGLE = 30.0
N_TRAIN = 55_000

# Valeurs utilisées quand un run ne les précise pas.
DEFAULTS = {
    "variant": "direct",  # "direct" ou "residual"
    "alpha": 1.0,  # poids du terme équivariant (méthode equivariant)
    "detach": True,  # figer la cible tournée (méthode equivariant)
    "epochs": 5,
    "lr": 1e-3,
    "batch_size": 128,
    "seed": 0,
    "final_test": False,
}

# Les runs à lancer. "method" est obligatoire : "supervised", "consistency"
# ou "equivariant". Le reste remplace les valeurs de DEFAULTS.
RUNS = [
    {
        "method": "supervised",
        "variant": "residual",
        "epochs": 20,
        "lr": 3e-3,
        "seed": s,
        "final_test": True,
    }
    for s in (0, 1, 2)
] + [
    {
        "method": "consistency",
        "variant": "direct",
        "epochs": 20,
        "lr": 3e-3,
        "seed": s,
        "final_test": True,
    }
    for s in (0, 1, 2)
]


def split_errors(model, x, mask, batch_size=500):
    """(MSE dans la partie mesurée, MSE dans le cône), moyennées sur x."""
    model.eval()
    measured = 0.0
    missing = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            m, c = split_mse(model(measure(xb, mask)), xb, mask)
            measured += float(m) * xb.shape[0]
            missing += float(c) * xb.shape[0]
    return measured / x.shape[0], missing / x.shape[0]


def run_name(cfg):
    name = f"{cfg['method']}_{cfg['variant']}"

    if cfg["method"] == "equivariant":
        detach_label = "detach" if cfg["detach"] else "nodetach"
        name += f"_a{cfg['alpha']:g}_{detach_label}"

    name += f"_e{cfg['epochs']}_lr{cfg['lr']:g}"
    if cfg["final_test"]:
        name += "_final"
    return f"{name}_seed{cfg['seed']}"


def run(cfg, x_train, x_val, mask):
    if cfg["method"] == "supervised":
        loss_fn = supervised_loss
    elif cfg["method"] == "consistency":
        loss_fn = consistency_loss
    else:
        loss_fn = make_equivariant_loss(alpha=cfg["alpha"], detach=cfg["detach"])

    torch.manual_seed(cfg["seed"])
    model = ConvNet(residual=cfg["variant"] == "residual")

    baseline = psnr(measure(x_val, mask), x_val)
    epoch0 = evaluate(model, x_val, mask)

    start = time.perf_counter()
    history = train(
        model,
        loss_fn,
        x_train,
        x_val,
        mask,
        cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    seconds = time.perf_counter() - start

    val_psnr = history["val_psnr"][-1]
    val_measured, val_missing = split_errors(model, x_val, mask)

    name = run_name(cfg)
    print(f"\n{name}")
    print(f"  PSNR de la baseline (validation) : {baseline:.2f} dB")
    print(f"  PSNR de validation avant entraînement : {epoch0:.2f} dB")
    print(
        f"  PSNR de validation final : {val_psnr:.2f} dB "
        f"({val_psnr - baseline:+.2f} dB par rapport à la baseline)"
    )
    print(
        f"  MSE validation : mesurée {val_measured:.2e}, dans le cône {val_missing:.2e}"
    )
    print(f"  temps : {seconds:.0f} s ({seconds / cfg['epochs']:.0f} s par époque)")

    result = dict(cfg)
    result.update(
        {
            "data_seed": DATA_SEED,
            "half_angle_deg": HALF_ANGLE,
            "n_train": N_TRAIN,
            "n_val": int(x_val.shape[0]),
            "baseline_val_psnr_db": float(baseline),
            "val_psnr_before_training_db": float(epoch0),
            "val_mse_measured": val_measured,
            "val_mse_missing": val_missing,
            "history": history,
            "seconds_total": seconds,
            "torch_version": torch.__version__,
        }
    )

    if cfg["final_test"]:
        x_test = load_mnist32(train=False, seed=TEST_SEED)
        result["test_psnr_db"] = float(evaluate(model, x_test, mask))
        result["test_baseline_psnr_db"] = float(psnr(measure(x_test, mask), x_test))
        print(
            f"  TEST : {result['test_psnr_db']:.2f} dB "
            f"(baseline {result['test_baseline_psnr_db']:.2f} dB)"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), runs_dir / f"{name}.pt")


if __name__ == "__main__":
    # Vérifier tout avant de charger les données : une faute de frappe dans RUNS
    # (par exemple "epoch" au lieu de "epochs") serait sinon ignorée sans message.
    configs = []
    for overrides in RUNS:
        assert "method" in overrides, overrides
        unknown = set(overrides) - set(DEFAULTS) - {"method"}
        assert not unknown, f"réglages inconnus : {unknown}"
        cfg = {**DEFAULTS, **overrides}  # les valeurs de overrides l'emportent
        assert cfg["method"] in ("supervised", "consistency", "equivariant"), cfg
        assert cfg["variant"] in ("direct", "residual"), cfg
        configs.append(cfg)

    x_all = load_mnist32(train=True, seed=DATA_SEED)
    x_train, x_val = x_all[:N_TRAIN], x_all[N_TRAIN:]
    mask = wedge_mask(32, HALF_ANGLE)

    for cfg in configs:
        run(cfg, x_train, x_val, mask)
