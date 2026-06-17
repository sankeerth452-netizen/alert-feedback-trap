"""Diagnostic: is the post-drift regime (t>=43) internally learnable?

D0: train 1-34            -> test 43-49   (reference: the known collapse)
D1: train 1-42            -> test 43-49   (do val-era labels help? = more pre-drift data)
D2: train 43-44 ONLY      -> test 45-49   (oracle adaptation, 2 steps of new labels)
D3: train 43-45 ONLY      -> test 46-49   (oracle adaptation, 3 steps)

If D2/D3 >> D0, adaptation with fresh labels works -> AEGIS v2 is viable.
If D2/D3 ~= D0, the post-drift illicit class is unlearnable from features
               -> the paper pivots to detection + fundamental-limits framing.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier

from src.data.load import FEATURE_COLS, load_dataframe
from src.eval.metrics import aggregate_metrics
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)


def run_setting(name, X, y, t, train_lo, train_hi, test_lo, test_hi, seed):
    tr = (t >= train_lo) & (t <= train_hi) & (y != -1)
    te = (t >= test_lo) & (t <= test_hi) & (y != -1)
    clf = LGBMClassifier(random_state=seed, **LGBM)
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[te])[:, 1]
    m = aggregate_metrics(y[te], prob)
    print(f"{name:>28} | train[{train_lo:2d},{train_hi:2d}] "
          f"({tr.sum():5d} lbl, {(y[tr]==1).sum():4d} illicit) "
          f"-> test[{test_lo},{test_hi}] : "
          f"F1={m['f1_illicit']:.4f}  P={m['precision_illicit']:.4f}  "
          f"R={m['recall_illicit']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    return {"setting": name, **m}


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X = df[FEATURE_COLS].values
    y = df["label"].values
    t = df["time_step"].values

    print("Labeled illicit per post-drift step:")
    for step in range(43, 50):
        m = (t == step) & (y != -1)
        print(f"  t={step}: {m.sum():4d} labeled, {(y[m]==1).sum():3d} illicit")
    print()

    rows = [
        run_setting("D0 static (collapse ref)", X, y, t, 1, 34, 43, 49, cfg["seed"]),
        run_setting("D1 static + val labels",   X, y, t, 1, 42, 43, 49, cfg["seed"]),
        run_setting("D2 oracle adapt (2 steps)", X, y, t, 43, 44, 45, 49, cfg["seed"]),
        run_setting("D3 oracle adapt (3 steps)", X, y, t, 43, 45, 46, 49, cfg["seed"]),
    ]
    pd.DataFrame(rows).to_csv(
        Path(cfg["paths"]["results_metrics"]) / "diagnostics.csv", index=False)


if __name__ == "__main__":
    main()