"""AEGIS-Meta: competence-gated meta-ensemble over heterogeneous learners.

Experts: static LightGBM (1-34) | online SRP (continuously updated from
revealed labels) | alarm-spawned adaptive LightGBM.
Gate: softmax over per-expert competence (EMA of AP on revealed labels).
Regime: strict budget feedback (top-k + epsilon exploration, delay 1).

Run:  python experiments/run_aegis_meta.py --seeds 42 43 44
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
from lightgbm import LGBMClassifier
from river import ensemble
from sklearn.metrics import average_precision_score

from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
BUDGET_K, LABEL_DELAY, EPSILON = 50, 1, 0.2
ALARM_Z, MIN_POS, MIN_NEG, ADAPT_WINDOW = 2.0, 5, 5, 8
COMP_TAU, COMP_EMA = 0.10, 0.6


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def run_seed(X, y, t, train_end, psi_z, seed):
    rng = np.random.default_rng(seed)
    Xd_all = None  # lazy dict cache for river

    static = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= train_end) & (y != -1)
    static.fit(X[bm], y[bm])

    srp = ensemble.SRPClassifier(n_models=10, seed=seed)
    order = np.argsort(t, kind="stable")
    for i in order:
        if t[i] <= train_end and y[i] != -1:
            srp.learn_one(dict(zip(FEATURE_COLS, X[i])), int(y[i]))

    adaptive = None
    comp = {"static": 0.5, "srp": 0.5, "adp": 0.5}
    revealed, consumed = {}, set()
    rows = []

    for s in range(train_end + 1, 50):
        # ---- delayed learning: SRP consumes revealed labels ----
        for q in [q for q in revealed if q <= s - LABEL_DELAY and q not in consumed]:
            for i in revealed[q]:
                srp.learn_one(dict(zip(FEATURE_COLS, X[i])), int(y[i]))
            consumed.add(q)

        idx_s = np.where((t == s) & (y != -1))[0]
        Xs, ys = X[idx_s], y[idx_s]
        Xs_d = [dict(zip(FEATURE_COLS, r)) for r in Xs]

        # ---- expert scores ----
        scores = {"static": static.predict_proba(Xs)[:, 1],
                  "srp": np.array([srp.predict_proba_one(d).get(1, 0.0)
                                   for d in Xs_d])}
        if adaptive is not None:
            scores["adp"] = adaptive.predict_proba(Xs)[:, 1]

        names = list(scores)
        logits = np.array([comp[n] for n in names]) / COMP_TAU
        logits -= logits.max()
        w = np.exp(logits); w /= w.sum()
        prob = sum(wi * scores[n] for wi, n in zip(w, names))

        # ---- metrics ----
        k = min(BUDGET_K, len(idx_s))
        top = np.argsort(prob)[::-1][:k]
        rows.append({"seed": seed, "time_step": s,
                     "pr_auc": safe_ap(ys, prob),
                     "precision_at_k": float(np.mean(ys[top] == 1)),
                     "illicit_caught": int(np.sum(ys[top] == 1)),
                     "n_illicit": int((ys == 1).sum()),
                     **{f"w_{n}": round(float(wi), 3)
                        for wi, n in zip(w, names)}})

        # ---- investigate: exploit + explore ----
        n_explore = int(EPSILON * k)
        exploit = top[: k - n_explore]
        rest = np.setdiff1d(np.arange(len(idx_s)), exploit)
        explore = rng.choice(rest, size=min(n_explore, len(rest)), replace=False)
        chosen = np.concatenate([exploit, explore])
        revealed[s] = idx_s[chosen]

        # ---- competence update on this step's revealed labels ----
        yl = ys[chosen]
        if (yl == 1).any() and (yl == 0).any():
            for n in names:
                ap = safe_ap(yl, scores[n][chosen])
                comp[n] = COMP_EMA * comp[n] + (1 - COMP_EMA) * ap

        # ---- alarm-triggered adaptive expert (delayed labels) ----
        if psi_z[s] >= ALARM_Z or adaptive is not None:
            avail = [q for q in revealed
                     if s - LABEL_DELAY - ADAPT_WINDOW < q <= s - LABEL_DELAY]
            if avail:
                pool = np.concatenate([revealed[q] for q in avail])
                yp = y[pool]
                if (yp == 1).sum() >= MIN_POS and (yp == 0).sum() >= MIN_NEG:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM)
                    adaptive.fit(X[pool], yp)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, train_end + 1)])
    psi_z = {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}

    all_rows = []
    for seed in args.seeds:
        print(f"\n=== AEGIS-Meta | seed {seed} ===")
        set_seed(seed)
        rows = run_seed(X, y, t, train_end, psi_z, seed)
        all_rows += rows
        for r in rows:
            pa = r["pr_auc"]
            ws = " ".join(f"{k[2:]}={v}" for k, v in r.items() if k.startswith("w_"))
            print(f"t={r['time_step']:2d} | PR-AUC={pa if pa == pa else float('nan'):.3f} "
                  f"| P@{BUDGET_K}={r['precision_at_k']:.3f} "
                  f"| caught {r['illicit_caught']:2d}/{r['n_illicit']:2d} | {ws}")

    res = pd.DataFrame(all_rows)
    res.to_csv(metrics_dir / "aegis_meta_results.csv", index=False)
    post = res[res.time_step >= 43]
    per_seed = post.groupby("seed").agg(mean_pr_auc=("pr_auc", "mean"),
                                        total_caught=("illicit_caught", "sum"))
    print("\n===== AEGIS-Meta POST-DRIFT (43-49), per seed =====")
    print(per_seed.to_string())
    print(f"\nmean caught = {per_seed.total_caught.mean():.1f} "
          f"± {per_seed.total_caught.std():.1f} | "
          f"mean PR-AUC = {per_seed.mean_pr_auc.mean():.4f} "
          f"± {per_seed.mean_pr_auc.std():.4f}")


if __name__ == "__main__":
    main()