"""Aggregate the final runs into a Markdown table (results/summary.md).

A final run is a file results/<config>_final_seed<k>.json. Runs of the same
configuration (same name, different seeds) are grouped.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, std


def main():
    groups = defaultdict(list)
    for path in sorted((ROOT / "results").glob("*_final_seed*.json")):
        groups[path.stem.rsplit("_seed", 1)[0]].append(json.loads(path.read_text()))

    if not groups:
        print("No final run found in results/ (files *_final_seed*.json).")
        return

    all_runs = [run for runs in groups.values() for run in runs]
    baseline = all_runs[0]["test_baseline_psnr_db"]

    lines = [
        "| Configuration | Seeds | Test PSNR (dB) | Gain vs baseline (dB) | "
        "Val MSE measured | Val MSE in cone | s/epoch |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, runs in groups.items():
        psnr_m, psnr_s = mean_std([r["test_psnr_db"] for r in runs])
        gain_m, gain_s = mean_std(
            [r["test_psnr_db"] - r["test_baseline_psnr_db"] for r in runs]
        )
        measured = statistics.mean(r["val_mse_measured"] for r in runs)
        missing = statistics.mean(r["val_mse_missing"] for r in runs)
        sec = statistics.mean(r["seconds_total"] / r["epochs"] for r in runs)
        lines.append(
            f"| {name} | {len(runs)} | {psnr_m:.2f} ± {psnr_s:.2f} | "
            f"{gain_m:+.2f} ± {gain_s:.2f} | {measured:.2e} | {missing:.2e} | {sec:.0f} |"
        )

    lines.append("")
    lines.append(
        f"Baseline test PSNR (x_hat = y): {baseline:.2f} dB. "
        "± is the sample standard deviation (n-1) over seeds."
    )

    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "summary.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
