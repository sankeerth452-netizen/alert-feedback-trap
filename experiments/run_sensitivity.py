"""Sensitivity of detect-and-adapt to label DELAY and label NOISE.

Sweeps:
  delay in {1, 2, 4} steps   (how late investigated labels arrive)
  noise in {0%, 10%, 25%}    (fraction of revealed labels flipped --
                              investigator error / ambiguous cases)

Regimes: adaptive_full (bulk feedback, PSI gate -- identical to
run_aegis_final) and aegis_cg (alert-only + exploration, competence gate),
both at k=50. Training and competence updates see the NOISY labels (that's
what the institution believes); evaluation metrics use ground truth.

Results APPEND to sensitivity_results.csv; reruns of the same
(mode, delay, noise, seed) override earlier rows.

Runs:
  python experiments/run_sensitivity.py --seeds 42 43 44 45 46
  python experiments/run_sensitivity.py --modes adaptive_full --seeds 42 43 44 45 46
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
BUDGET_K, ALARM_Z = 50, 2.0
MIN_POS, MIN_NEG, ADAPT_WINDOW = 5, 5, 8
EPSILON, COMP_TAU, COMP_EMA = 0.2, 0.10, 0.6
DELAYS = [1, 2, 4]
NOISES = [0.0, 0.10, 0.25]


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def run_sim(mode, delay, noise, X, y, t, train_end, psi_z, seed):
    rng = np.random.default_rng(seed)
    base = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= train_end) & (y != -1)
    base.fit(X[bm], y[bm])
    base_ref = X[t <= train_end]

    adaptive, adaptive_ref = None, None
    comp = {"base": 0.5, "adp": 0.5}
    revealed = {}              # step -> (indices, noisy_labels)
    caught_post, pr_post = 0, []

    for s in range(train_end + 1, 50):
        idx_s = np.where((t == s) & (y != -1))[0]
        Xs, ys = X[idx_s], y[idx_s]

        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None:
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            if mode == "aegis_cg":
                logits = np.array([comp["base"], comp["adp"]]) / COMP_TAU
            else:                                   # adaptive_full: PSI gate
                psis = np.array([mean_psi(base_ref, Xs),
                                 mean_psi(adaptive_ref, Xs)])
                logits = -psis / 0.05
            logits -= logits.max()
            w = np.exp(logits); w /= w.sum()
            prob = w[0] * p_base + w[1] * p_adp
        else:
            p_adp, prob = None, p_base

        kk = min(BUDGET_K, len(idx_s))
        top = np.argsort(prob)[::-1][:kk]
        if s >= 43:
            caught_post += int(np.sum(ys[top] == 1))
            pr_post.append(safe_ap(ys, prob))

        # ---- investigation -> noisy revealed labels ----
        if mode == "adaptive_full":
            chosen = np.arange(len(idx_s))
        else:                                   # aegis_cg
            n_x = int(EPSILON * kk)
            exploit = top[: kk - n_x]
            rest = np.setdiff1d(np.arange(len(idx_s)), exploit)
            explore = rng.choice(rest, size=min(n_x, len(rest)), replace=False)
            chosen = np.concatenate([exploit, explore])

        noisy = ys[chosen].copy()
        flips = rng.random(len(noisy)) < noise
        noisy[flips] = 1 - noisy[flips]
        revealed[s] = (idx_s[chosen], noisy)

        if mode == "aegis_cg" and adaptive is not None and \
           (noisy == 1).any() and (noisy == 0).any():
            for name, p in [("base", p_base), ("adp", p_adp)]:
                comp[name] = COMP_EMA * comp[name] + \
                             (1 - COMP_EMA) * safe_ap(noisy, p[chosen])

        # ---- alarm-triggered adaptation on delayed noisy labels ----
        if psi_z[s] >= ALARM_Z or adaptive is not None:
            avail = [q for q in revealed
                     if s - delay - ADAPT_WINDOW < q <= s - delay]
            if avail:
                pool_idx = np.concatenate([revealed[q][0] for q in avail])
                pool_lab = np.concatenate([revealed[q][1] for q in avail])
                if (pool_lab == 1).sum() >= MIN_POS and \
                   (pool_lab == 0).sum() >= MIN_NEG:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM)
                    adaptive.fit(X[pool_idx], pool_lab)
                    adaptive_ref = X[np.isin(t, avail)]

    return {"mode": mode, "delay": delay, "noise": noise, "seed": seed,
            "caught_post": caught_post,
            "mean_pr_auc_post": float(np.nanmean(pr_post))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--modes", nargs="*",
                        default=["adaptive_full", "aegis_cg"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])
    fig_dir = Path(cfg["paths"]["results_figures"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, train_end + 1)])
    psi_z = {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}

    rows, done = [], 0
    total = len(args.modes) * len(DELAYS) * len(NOISES) * len(args.seeds)
    for mode in args.modes:
        for delay in DELAYS:
            for noise in NOISES:
                for seed in args.seeds:
                    set_seed(seed)
                    r = run_sim(mode, delay, noise, X, y, t,
                                train_end, psi_z, seed)
                    rows.append(r); done += 1
                    print(f"[{done:3d}/{total}] {mode:>13} delay={delay} "
                          f"noise={noise:.2f} seed={seed} | "
                          f"caught={r['caught_post']:3d} "
                          f"| PR-AUC={r['mean_pr_auc_post']:.3f}")

    # ---- append, then aggregate from full deduped history ----
    out_csv = metrics_dir / "sensitivity_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, mode="a",
                              header=not out_csv.exists(), index=False)
    res = pd.read_csv(out_csv).drop_duplicates(
        subset=["mode", "delay", "noise", "seed"], keep="last")

    agg = res.groupby(["mode", "delay", "noise"]).agg(
        caught_mean=("caught_post", "mean"),
        caught_std=("caught_post", "std"),
        pr_auc_mean=("mean_pr_auc_post", "mean")).round(3)
    print("\n===== SENSITIVITY (post-drift, mean over seeds, all runs) =====")
    print(agg.to_string())

    # ---- Figure 9: caught vs delay, one line per noise level, per regime ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mode in zip(axes, ["adaptive_full", "aegis_cg"]):
        g = res[res["mode"] == mode]
        for noise in NOISES:
            gg = g[g["noise"] == noise].groupby("delay")["caught_post"].mean()
            if len(gg):
                ax.plot(gg.index, gg.values, marker="o",
                        label=f"noise={int(noise*100)}%")
        ax.set_title(mode); ax.set_xlabel("Label delay (steps)")
        ax.set_xticks(DELAYS)
    axes[0].set_ylabel("Illicit caught post-drift (of 169)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "fig9_sensitivity.pdf"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 9 -> {out}")


if __name__ == "__main__":
    main()