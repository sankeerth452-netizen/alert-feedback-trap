"""Compute the PSI drift signal for every time step vs the train reference.

Produces Figure 3: the drift alarm. We expect it to spike at t=43
WITHOUT using any labels -- i.e., the alarm works in production, where
labels arrive months late or never.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    fig_dir = Path(cfg["paths"]["results_figures"])
    metrics_dir = Path(cfg["paths"]["results_metrics"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X = df[FEATURE_COLS].values
    t = df["time_step"].values

    ref = X[t <= cfg["split"]["train_end"]]       # training-era distribution

    rows = []
    for step in sorted(df["time_step"].unique()):
        rows.append({"time_step": int(step),
                     "mean_psi": mean_psi(ref, X[t == step])})
        print(f"t={step:2d}  mean PSI = {rows[-1]['mean_psi']:.4f}")

    scan = pd.DataFrame(rows)
    scan.to_csv(metrics_dir / "drift_scan_psi.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scan["time_step"], scan["mean_psi"], marker="o",
            markersize=3.5, color="#c44e52", linewidth=1.6)
    ax.axvline(43, color="black", linestyle="--", linewidth=1.2,
               label="Dark market shutdown (t=43)")
    ax.axvspan(1, cfg["split"]["train_end"], alpha=0.08, color="grey",
               label="Train reference window")
    ax.axhline(0.25, color="orange", linestyle=":", linewidth=1.2,
               label="Major-shift threshold (PSI=0.25)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean PSI vs train reference")
    ax.set_title("Label-free drift alarm (Population Stability Index)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "fig3_drift_alarm.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 3 -> {out}")


if __name__ == "__main__":
    main()