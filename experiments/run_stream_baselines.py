"""Streaming/adaptive baselines (river) under our prequential protocol.

Models: ARF (Adaptive Random Forest, ADWIN-based), SRP (Streaming Random
Patches), HAT (Hoeffding Adaptive Tree).

Feedback regimes (mirror run_aegis_final exactly):
  full   : all labels of each step arrive with delay=1   (vs adaptive_full)
  budget : only top-k investigated alerts yield labels   (vs aegis_cg)

Protocol: pretrain on steps 1-34 (single time-ordered pass), then for each
step 35..49: predict first, then learn from labels revealed at delay.

Run:
  python experiments/run_stream_baselines.py                      # 1 seed, all
  python experiments/run_stream_baselines.py --models arf --seeds 42 43 44
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from river import ensemble, forest, tree
from sklearn.metrics import average_precision_score

from src.data.load import FEATURE_COLS, load_dataframe
from src.utils.seed import set_seed

BUDGET_K = 50
LABEL_DELAY = 1


def make_model(name, seed):
    if name == "arf":
        return forest.ARFClassifier(n_models=10, seed=seed)
    if name == "srp":
        return ensemble.SRPClassifier(n_models=10, seed=seed)
    if name == "hat":
        return tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    raise ValueError(name)


def to_dicts(X):
    return [dict(zip(FEATURE_COLS, row)) for row in X]


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def run(model_name, regime, X, y, t, train_end, seed):
    model = make_model(model_name, seed)

    # ---- pretrain: single time-ordered pass over labeled train data ----
    order = np.argsort(t, kind="stable")
    for i in order:
        if t[i] <= train_end and y[i] != -1:
            model.learn_one(dict(zip(FEATURE_COLS, X[i])), int(y[i]))

    pending = {}          # step -> list of (x_dict, label) to learn later
    rows = []
    for s in range(train_end + 1, 50):
        # ---- learn from labels revealed for step s - LABEL_DELAY ----
        for q in [q for q in list(pending) if q <= s - LABEL_DELAY]:
            for xd, yl in pending.pop(q):
                model.learn_one(xd, int(yl))

        # ---- predict current step (before seeing its labels) ----
        idx_s = np.where((t == s) & (y != -1))[0]
        Xd = to_dicts(X[idx_s])
        prob = np.array([model.predict_proba_one(xd).get(1, 0.0) for xd in Xd])
        ys = y[idx_s]

        k = min(BUDGET_K, len(idx_s))
        top = np.argsort(prob)[::-1][:k]
        rows.append({"model": model_name, "regime": regime, "seed": seed,
                     "time_step": s, "pr_auc": safe_ap(ys, prob),
                     "precision_at_k": float(np.mean(ys[top] == 1)),
                     "illicit_caught": int(np.sum(ys[top] == 1)),
                     "n_illicit": int((ys == 1).sum())})

        # ---- queue label revelation per regime ----
        if regime == "full":
            chosen = np.arange(len(idx_s))
        else:                                   # budget: investigated only
            chosen = top
        pending[s] = [(Xd[i], ys[i]) for i in chosen]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["arf", "srp", "hat"])
    parser.add_argument("--regimes", nargs="*", default=["full", "budget"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    all_rows = []
    for m in args.models:
        for regime in args.regimes:
            for seed in args.seeds:
                print(f"\n=== {m} | regime={regime} | seed={seed} ===")
                rows = run(m, regime, X, y, t, train_end, seed)
                all_rows += rows
                for r in rows:
                    pa = r["pr_auc"]
                    print(f"t={r['time_step']:2d} | "
                          f"PR-AUC={pa if pa == pa else float('nan'):.3f} | "
                          f"P@{BUDGET_K}={r['precision_at_k']:.3f} | "
                          f"caught {r['illicit_caught']:2d}/{r['n_illicit']:2d}")

    res = pd.DataFrame(all_rows)
    out = metrics_dir / "stream_baselines_results.csv"
    # append across invocations so per-model runs accumulate
    res.to_csv(out, mode="a", header=not out.exists(), index=False)

    post = res[res.time_step >= 43]
    agg = post.groupby(["model", "regime", "seed"]).agg(
        mean_pr_auc=("pr_auc", "mean"),
        total_caught=("illicit_caught", "sum")).reset_index()
    print("\n===== POST-DRIFT (43-49), per model/regime (across seeds) =====")
    print(agg.groupby(["model", "regime"]).agg(
        pr_auc_mean=("mean_pr_auc", "mean"),
        pr_auc_std=("mean_pr_auc", "std"),
        caught_mean=("total_caught", "mean"),
        caught_std=("total_caught", "std")).round(4).to_string())


if __name__ == "__main__":
    main()