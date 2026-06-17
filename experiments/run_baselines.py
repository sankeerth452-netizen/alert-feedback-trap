"""Train tabular baselines on the temporal split; save metrics + Figure 2.

Run from anywhere:
    python experiments/run_baselines.py                      # all models
    python experiments/run_baselines.py --models lightgbm    # subset, fresh process
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data.load import FEATURE_COLS, load_dataframe
from src.data.splits import temporal_masks
from src.eval.metrics import aggregate_metrics, per_timestep_f1
from src.models.tabular import build_model
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of models to run, e.g. --models lightgbm")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    models_cfg = yaml.safe_load(open("configs/baselines.yaml"))["models"]
    if args.models:
        unknown = set(args.models) - set(models_cfg)
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. "
                             f"Available: {list(models_cfg)}")
        models_cfg = {k: v for k, v in models_cfg.items() if k in args.models}
    set_seed(cfg["seed"])

    metrics_dir = Path(cfg["paths"]["results_metrics"])
    fig_dir = Path(cfg["paths"]["results_figures"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(cfg["paths"]["raw_dir"])
    train_mask, val_mask, test_mask = temporal_masks(df, **cfg["split"])

    X = df[FEATURE_COLS].values
    y = df["label"].values
    t = df["time_step"].values

    eval_mask = val_mask | test_mask  # steps 35-49, for the drift curve

    summary_rows, curves = [], {}
    for name, params in models_cfg.items():
        print(f"\n--- Training {name} ---")
        model = build_model(name, params, cfg["seed"])
        model.fit(X[train_mask], y[train_mask])

        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            prob = model.predict_proba(X[mask])[:, 1]
            m = aggregate_metrics(y[mask], prob)
            summary_rows.append({"model": name, "split": split_name, **m})
            print(f"{split_name}: {m}")

        prob_eval = model.predict_proba(X[eval_mask])[:, 1]
        curve = per_timestep_f1(y[eval_mask], prob_eval, t[eval_mask])
        curve.to_csv(metrics_dir / f"curve_{name}.csv", index=False)
        curves[name] = curve

    suffix = "_".join(models_cfg.keys()) if args.models else "all"

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / f"baselines_summary_{suffix}.csv", index=False)
    with open(metrics_dir / f"baselines_config_used_{suffix}.json", "w") as f:
        json.dump(models_cfg, f, indent=2)

    print("\n================ SUMMARY ================")
    print(summary.to_string(index=False))

    # ---- Figure 2: per-time-step F1 (the collapse curve) ----
    # When run with a subset, curves for missing models are read from disk
    # (saved by earlier runs), so Figure 2 always shows every available model.
    all_model_names = list(yaml.safe_load(open("configs/baselines.yaml"))["models"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plotted = 0
    for name in all_model_names:
        if name in curves:
            curve = curves[name]
        elif (metrics_dir / f"curve_{name}.csv").exists():
            curve = pd.read_csv(metrics_dir / f"curve_{name}.csv")
        else:
            continue
        ax.plot(curve["time_step"], curve["f1_illicit"], marker="o",
                markersize=3.5, linewidth=1.5, label=name)
        plotted += 1
    ax.axvline(43, color="black", linestyle="--", linewidth=1.2,
               label="Dark market shutdown (t=43)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("F1 (illicit class)")
    ax.set_title("Baseline performance over time: the drift collapse")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "fig2_baseline_collapse.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 2 ({plotted} models) -> {out}")


if __name__ == "__main__":
    main()