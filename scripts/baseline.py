from missing_wedge_ei.metrics import psnr, split_mse
from missing_wedge_ei.forward_op import wedge_mask, measure
from missing_wedge_ei.data import load_mnist32
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str: #The idea is to be able to retrieve what version gave the result we are saving.
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


#Baseline gives the error between original one and measured one without further reconstruction. It's a lower bound.

def main():
    SEED = 1
    HALF_ANGLE = 30.0
    MASK = wedge_mask(32, HALF_ANGLE)

    x = load_mnist32(train=False, seed=SEED, rotate=True)
    y = measure(x, MASK)
    print("PSNR baseline (dB) :")
    psnr_value = psnr(y, x)
    print(psnr_value)  # should be around 17.5 dB
    print("MSE baseline (measured, missing) :")
    measured, missing = split_mse(y, x, MASK)
    print(measured, missing, measured + missing)

    result = {
    "method": "baseline",
    "split": "test",
    "n_images": int(x.shape[0]),
    "data_seed": SEED,                     # la graine passée à load_mnist32
    "half_angle_deg": HALF_ANGLE,
    "psnr_db": float(psnr_value),
    "mse_measured": float(measured),
    "mse_missing": float(missing),
    "torch_version": torch.__version__,
    "git_commit": git_commit(),
    }
    out = ROOT / "results" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("résultats sauvegardés dans", out)

if __name__ == "__main__":
    main()

