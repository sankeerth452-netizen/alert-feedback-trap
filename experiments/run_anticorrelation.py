"""Characterize WHY greedy investigation starves post-drift on Elliptic.

We had hypothesized anti-correlation (negative score-label correlation). The
data refuted that: post-drift Spearman stays weakly POSITIVE. The true picture
is sharper -- the detector keeps a faint RANK signal but its CALIBRATION and
TOP-K PRECISION collapse: the top-50 alerts (what greedy investigation actually
consumes) contain near-zero illicit, at several steps performing at or BELOW
the random rate. That is what starves the feedback loop, even without sign-flip.

Run:  python experiments/run_anticorrelation.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from src.data.load import FEATURE_COLS, load_dataframe
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
K = 50


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])
    train_end = cfg["split"]["train_end"]
    mdir = Path(cfg["paths"]["results_metrics"])
    fdir = Path(cfg["paths"]["results_figures"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    bm = (t <= train_end) & (y != -1)
    model = LGBMClassifier(random_state=cfg["seed"], **LGBM).fit(X[bm], y[bm])

    rows = []
    for s in range(train_end + 1, 50):
        m = (t == s) & (y != -1)
        Xs, ys = X[m], y[m]
        if len(ys) < 10 or ys.sum() == 0:
            continue
        p = model.predict_proba(Xs)[:, 1]
        rho = spearmanr(p, ys).statistic
        base_rate = ys.mean()
        pr = average_precision_score(ys, p)
        top = np.argsort(p)[::-1][:min(K, len(ys))]
        prec_at_k = (ys[top] == 1).mean()
        rows.append({
            "time_step": s, "n": len(ys), "base_rate": round(base_rate, 4),
            "spearman": round(float(rho), 4),
            "pr_auc": round(float(pr), 4),
            "pr_auc_vs_random": round(float(pr - base_rate), 4),
            "prec_at_50": round(float(prec_at_k), 4),
            "prec_vs_random": round(float(prec_at_k - base_rate), 4),
            "regime": "post-drift" if s >= 43 else "pre-drift",
        })

    res = pd.DataFrame(rows)
    res.to_csv(mdir / "anticorrelation.csv", index=False)
    print(res.to_string(index=False))

    post = res[res.time_step >= 43]
    n_neg_rho = (post.spearman < 0).sum()
    n_prec_below = (post.prec_vs_random < 0).sum()
    print(f"\nPost-drift steps with NEGATIVE score-label correlation: "
          f"{n_neg_rho}/{len(post)}  (hypothesis of anti-correlation: "
          f"{'supported' if n_neg_rho > len(post)/2 else 'REFUTED'})")
    print(f"Post-drift steps with top-50 precision AT/BELOW random: "
          f"{n_prec_below}/{len(post)}")
    print(f"Mean post-drift Spearman: {post.spearman.mean():.4f}  "
          f"(weak positive: faint rank signal persists)")
    print(f"Mean post-drift precision@50 minus random: "
          f"{post.prec_vs_random.mean():.4f}")
    print("Interpretation: the detector keeps a faint RANK signal, but its "
          "TOP-50 precision collapses to/below random. Greedy investigation "
          "consumes the top-k, so it harvests ~no positives and the feedback "
          "loop starves -- the trap mechanism, without any sign-flip.")

    # ---- Figure 10: two panels -- rank signal persists vs top-k precision collapses ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    specs = [
        ("spearman", "Rank signal persists (weakly positive)",
         "Spearman corr(score, label)"),
        ("prec_vs_random", "but top-50 precision collapses to / below random",
         "precision@50  \u2212  base rate"),
    ]
    for ax, (col, ttl, ylab) in zip(axes, specs):
        colors = ["#4878a8" if r < 43 else "#c44e52" for r in res.time_step]
        ax.bar(res.time_step, res[col], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(42.5, color="black", linestyle="--", linewidth=1.2,
                   label="Dark market shutdown")
        ax.set_xlabel("Time step")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = fdir / "fig10_topk_collapse.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 10 -> {out}")


if __name__ == "__main__":
    main()