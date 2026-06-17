"""AEGIS final experiment: detect-and-adapt with competence-aware gating.

Modes:
  static           : frozen base model (deployed reality)
  adaptive_full    : bulk delayed labels, PSI gate            (upper feedback regime)
  adaptive_budget  : top-k-only labels, PSI gate              (the feedback trap)
  aegis_cg         : top-k-only labels + COMPETENCE gate + epsilon exploration
                     (the proposed system under strict realism)

Run from anywhere:
    python experiments/run_aegis_final.py                       # 1 seed
    python experiments/run_aegis_final.py --seeds 42 43 44 45 46
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

BUDGET_K = 50
LABEL_DELAY = 1
ALARM_Z = 2.0
MIN_POS, MIN_NEG = 5, 5      # min labels of each class to (re)train adaptive expert
ADAPT_WINDOW = 8
EPSILON = 0.2                # fraction of budget spent on exploration
COMP_TAU = 0.10              # competence-gate softmax temperature
COMP_EMA = 0.6               # EMA factor for competence updates


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def run_mode(mode, X, y, t, train_end, psi_z, seed):
    rng = np.random.default_rng(seed)
    base = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= train_end) & (y != -1)
    base.fit(X[bm], y[bm])
    base_ref = X[t <= train_end]

    adaptive, adaptive_ref = None, None
    comp = {"base": 0.5, "adp": 0.5}      # competence EMA, neutral start
    revealed = {}
    rows = []

    for s in range(train_end + 1, 50):
        idx_s = np.where((t == s) & (y != -1))[0]
        Xs, ys = X[idx_s], y[idx_s]

        # ---- score ----
        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None:
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            if mode == "aegis_cg":
                logits = np.array([comp["base"], comp["adp"]]) / COMP_TAU
            else:                                   # PSI gate
                psis = np.array([mean_psi(base_ref, Xs),
                                 mean_psi(adaptive_ref, Xs)])
                logits = -psis / 0.05
            logits -= logits.max()
            w = np.exp(logits); w /= w.sum()
            prob = w[0] * p_base + w[1] * p_adp
        else:
            w, p_adp = np.array([1.0, 0.0]), None
            prob = p_base

        # ---- metrics ----
        k = min(BUDGET_K, len(idx_s))
        top = np.argsort(prob)[::-1][:k]
        rows.append({"mode": mode, "seed": seed, "time_step": s,
                     "pr_auc": safe_ap(ys, prob),
                     "precision_at_k": float(np.mean(ys[top] == 1)),
                     "illicit_caught": int(np.sum(ys[top] == 1)),
                     "n_illicit": int((ys == 1).sum()),
                     "w_adaptive": round(float(w[-1]), 3)})

        # ---- investigation -> revealed labels ----
        if mode == "adaptive_full":
            revealed[s] = idx_s
        elif mode in ("adaptive_budget", "aegis_cg"):
            if mode == "aegis_cg":
                n_explore = int(EPSILON * k)
                exploit = top[: k - n_explore]
                rest = np.setdiff1d(np.arange(len(idx_s)), exploit)
                explore = rng.choice(rest, size=min(n_explore, len(rest)),
                                     replace=False)
                chosen = np.concatenate([exploit, explore])
            else:
                chosen = top
            revealed[s] = idx_s[chosen]

            # ---- competence update from revealed labels of THIS step ----
            if mode == "aegis_cg" and adaptive is not None:
                yl = ys[chosen]
                if (yl == 1).any() and (yl == 0).any():
                    ap_b = safe_ap(yl, p_base[chosen])
                    ap_a = safe_ap(yl, p_adp[chosen])
                    comp["base"] = COMP_EMA * comp["base"] + (1 - COMP_EMA) * ap_b
                    comp["adp"] = COMP_EMA * comp["adp"] + (1 - COMP_EMA) * ap_a

        # ---- alarm-triggered adaptation on delayed labels ----
        if mode != "static" and (psi_z[s] >= ALARM_Z or adaptive is not None):
            avail = [q for q in revealed
                     if s - LABEL_DELAY - ADAPT_WINDOW < q <= s - LABEL_DELAY]
            if avail:
                pool = np.concatenate([revealed[q] for q in avail])
                yp = y[pool]
                if (yp == 1).sum() >= MIN_POS and (yp == 0).sum() >= MIN_NEG:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM)
                    adaptive.fit(X[pool], yp)
                    adaptive_ref = X[np.isin(t, avail)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])
    fig_dir = Path(cfg["paths"]["results_figures"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    # ---- calibrated PSI alarm (deterministic, computed once) ----
    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, train_end + 1)])
    psi_z = {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}

    all_rows = []
    for seed in args.seeds:
        set_seed(seed)
        for mode in ["static", "adaptive_full", "adaptive_budget", "aegis_cg"]:
            print(f"\n=== seed {seed} | mode: {mode} ===")
            rows = run_mode(mode, X, y, t, train_end, psi_z, seed)
            all_rows += rows
            for r in rows:
                pa = r["pr_auc"]
                print(f"t={r['time_step']:2d} | "
                      f"PR-AUC={pa if pa == pa else float('nan'):.3f} "
                      f"| P@{BUDGET_K}={r['precision_at_k']:.3f} "
                      f"| caught {r['illicit_caught']:2d}/{r['n_illicit']:2d} "
                      f"| w_adp={r['w_adaptive']:.2f}")

    res = pd.DataFrame(all_rows)
    res.to_csv(metrics_dir / "aegis_final_results.csv", index=False)

    # ---- aggregate: per-seed totals, then mean ± std across seeds ----
    post = res[res.time_step >= 43]
    per_seed = post.groupby(["mode", "seed"]).agg(
        mean_pr_auc=("pr_auc", "mean"),
        mean_p_at_k=("precision_at_k", "mean"),
        total_caught=("illicit_caught", "sum")).reset_index()
    agg = per_seed.groupby("mode").agg(
        pr_auc_mean=("mean_pr_auc", "mean"),
        pr_auc_std=("mean_pr_auc", "std"),
        p_at_k_mean=("mean_p_at_k", "mean"),
        caught_mean=("total_caught", "mean"),
        caught_std=("total_caught", "std")).round(4)
    print("\n===== POST-DRIFT (43-49) AGGREGATE, mean ± std over "
          f"{len(args.seeds)} seed(s) =====")
    print(agg.to_string())
    agg.to_csv(metrics_dir / "aegis_final_aggregate.csv")

    # ---- Figure 5: seed-averaged PR-AUC per step, with min-max band ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mode, g in res.groupby("mode"):
        gm = g.groupby("time_step")["pr_auc"].agg(["mean", "min", "max"]).reset_index()
        ax.plot(gm["time_step"], gm["mean"], marker="o", markersize=4,
                linewidth=1.6, label=mode)
        if len(args.seeds) > 1:
            ax.fill_between(gm["time_step"], gm["min"], gm["max"], alpha=0.15)
    ax.axvline(43, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time step"); ax.set_ylabel("PR-AUC (illicit)")
    ax.set_title("Static vs detect-and-adapt under three feedback regimes "
                 f"({len(args.seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "fig5_final_pr_auc.pdf"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 5 -> {out}")


if __name__ == "__main__":
    main()