"""AEGIS v2: sequential detect-and-adapt evaluation (prequential).

Simulates real AML deployment over steps 35..49, one step at a time:
  1. Score today's transactions with the current system.
  2. Investigators review the top-k alerts (budget k per step).
  3. Labels become available with a 1-step delay.
  4. If the calibrated PSI alarm is firing, (re)train an adaptive expert
     on recently available labels; a PSI gate blends base + adaptive.

Modes:
  static          : base model only (the collapse, as deployed reality)
  adaptive_full   : all labels of past steps arrive after delay (bulk feedback)
  adaptive_budget : ONLY investigated top-k alerts get labels (strict realism)

Run from anywhere:  python experiments/run_aegis_adaptive.py
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)

BUDGET_K = 50          # alerts investigated per step
LABEL_DELAY = 1        # steps until investigated/bulk labels arrive
ALARM_Z = 2.0          # PSI z-score that triggers adaptation
MIN_ILLICIT = 10       # min confirmed illicit labels before spawning expert
ADAPT_WINDOW = 8       # adaptive expert trains on labels from last W steps


def precision_at_k(y_true, prob, k):
    k = min(k, len(prob))
    idx = np.argsort(prob)[::-1][:k]
    return float(np.mean(y_true[idx] == 1)), int(np.sum(y_true[idx] == 1))


def safe_pr_auc(y_true, prob):
    return float(average_precision_score(y_true, prob)) if (y_true == 1).any() else np.nan


def run_mode(mode, X, y, t, train_end, psi_z, seed):
    base = LGBMClassifier(random_state=seed, **LGBM)
    base_mask = (t <= train_end) & (y != -1)
    base.fit(X[base_mask], y[base_mask])
    base_ref = X[t <= train_end]

    adaptive, adaptive_ref = None, None
    # revealed[step] = array of node indices whose labels are known
    revealed = {}
    rows = []

    for s in range(train_end + 1, 50):
        m = (t == s) & (y != -1)
        idx_s = np.where(m)[0]
        Xs, ys = X[idx_s], y[idx_s]

        # ---- 1. Score with current system (PSI-gated blend) ----
        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None:
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            psis = np.array([mean_psi(base_ref, Xs), mean_psi(adaptive_ref, Xs)])
            w = np.exp(-psis / 0.05)
            w /= w.sum()
            prob = w[0] * p_base + w[1] * p_adp
        else:
            w = np.array([1.0, 0.0])
            prob = p_base

        # ---- 2. Metrics ----
        p_at_k, caught = precision_at_k(ys, prob, BUDGET_K)
        rows.append({"mode": mode, "time_step": s,
                     "pr_auc": safe_pr_auc(ys, prob),
                     "precision_at_k": p_at_k, "illicit_caught": caught,
                     "n_illicit": int((ys == 1).sum()),
                     "w_adaptive": round(float(w[-1]), 3),
                     "psi_z": round(float(psi_z[s]), 2)})

        # ---- 3. Investigation -> label revelation ----
        if mode == "adaptive_budget":
            top = idx_s[np.argsort(prob)[::-1][:min(BUDGET_K, len(idx_s))]]
            revealed[s] = top
        elif mode == "adaptive_full":
            revealed[s] = idx_s            # bulk feedback (all labeled nodes)

        # ---- 4. Alarm-triggered adaptation (delayed labels only) ----
        if mode != "static" and psi_z[s] >= ALARM_Z or \
           (mode != "static" and adaptive is not None):
            avail_steps = [q for q in revealed
                           if q <= s - LABEL_DELAY and q > s - LABEL_DELAY - ADAPT_WINDOW]
            if avail_steps:
                pool = np.concatenate([revealed[q] for q in avail_steps])
                yp = y[pool]
                if (yp == 1).sum() >= MIN_ILLICIT and (yp == 0).sum() >= MIN_ILLICIT:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM)
                    adaptive.fit(X[pool], yp)
                    adaptive_ref = X[np.isin(t, avail_steps)]
    return rows


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])
    fig_dir = Path(cfg["paths"]["results_figures"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X = df[FEATURE_COLS].values
    y = df["label"].values
    t = df["time_step"].values

    # ---- Calibrated PSI alarm, computed inline (label-free) ----
    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    train_vals = np.array([psi[s] for s in range(1, train_end + 1)])
    mu, sd = train_vals.mean(), train_vals.std(ddof=1)
    psi_z = {s: (psi[s] - mu) / sd for s in psi}
    print(f"PSI baseline mean={mu:.4f} std={sd:.4f}; "
          f"alarm steps (z>={ALARM_Z}): "
          f"{[s for s in range(train_end+1,50) if psi_z[s] >= ALARM_Z]}")

    all_rows = []
    for mode in ["static", "adaptive_full", "adaptive_budget"]:
        print(f"\n=== mode: {mode} ===")
        rows = run_mode(mode, X, y, t, train_end, psi_z, cfg["seed"])
        all_rows += rows
        for r in rows:
            print(f"t={r['time_step']:2d} | PR-AUC={r['pr_auc'] if not np.isnan(r['pr_auc']) else float('nan'):.3f} "
                  f"| P@{BUDGET_K}={r['precision_at_k']:.3f} "
                  f"| caught {r['illicit_caught']:2d}/{r['n_illicit']:2d} "
                  f"| w_adp={r['w_adaptive']:.2f} | psi_z={r['psi_z']}")

    res = pd.DataFrame(all_rows)
    res.to_csv(metrics_dir / "aegis_adaptive_results.csv", index=False)

    # ---- Aggregate over the post-drift region (43-49) ----
    print("\n========== POST-DRIFT (43-49) AGGREGATE ==========")
    post = res[res.time_step >= 43]
    agg = post.groupby("mode").agg(
        mean_pr_auc=("pr_auc", "mean"),
        mean_p_at_k=("precision_at_k", "mean"),
        total_caught=("illicit_caught", "sum"),
        total_illicit=("n_illicit", "sum"),
    ).round(4)
    print(agg.to_string())
    agg.to_csv(metrics_dir / "aegis_adaptive_aggregate.csv")

    # ---- Figure 5: PR-AUC per step, three modes ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mode, g in res.groupby("mode"):
        ax.plot(g["time_step"], g["pr_auc"], marker="o", markersize=4,
                linewidth=1.6, label=mode)
    ax.axvline(43, color="black", linestyle="--", linewidth=1.2,
               label="Dark market shutdown (t=43)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("PR-AUC (illicit)")
    ax.set_title(f"AEGIS detect-and-adapt vs static deployment "
                 f"(budget k={BUDGET_K}/step, delay={LABEL_DELAY})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "fig5_adaptive_pr_auc.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 5 -> {out}")


if __name__ == "__main__":
    main()