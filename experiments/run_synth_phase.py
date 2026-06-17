"""Phase-diagram test of the ignition theory on the synthetic stream.

Sweep: theta (drift magnitude) x base_rate (rarity) x budget k x seed.
Detectors:
  lgbm     : LightGBM -- same model/loop as Elliptic (GENERALIZATION claim)
  logistic : logistic regression -- lets us measure the boundary's angle to
             the true post-drift concept (MECHANISM claim; geometry of the trap)

Pre-registered predictions (see chat): (1) collapse grows with theta, going
anti-correlated near pi/2; (2) critical budget k* rises with theta, falls with
base_rate; (3) bimodal ignite/trap near k*; (4) exploration shifts k* left.

Run:  python experiments/run_synth_phase.py --seeds 42 43 44 45 46
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.synth.generator import (generate_stream, concept_angle,
                                  T_SHUTDOWN, T as SYNTH_T)
from src.utils.seed import set_seed

LGBM = dict(n_estimators=300, num_leaves=48, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
ALARM_PERIODS = list(range(T_SHUTDOWN, SYNTH_T))   # adaptation allowed post-t*
MIN_POS, MIN_NEG, ADAPT_WINDOW, LABEL_DELAY = 5, 5, 6, 1
EPSILON, COMP_TAU, COMP_EMA = 0.2, 0.10, 0.6

THETAS = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
RATES = [0.01, 0.025, 0.05]
BUDGETS = [25, 50, 100]


def safe_ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if (y == 1).any() and (y == 0).any() else np.nan


def fit(kind, X, y, seed):
    if kind == "lgbm":
        return LGBMClassifier(random_state=seed, **LGBM).fit(X, y)
    return LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=seed).fit(X, y)


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def run(kind, explore, k, stream, seed):
    """One prequential pass. explore=True -> eps-exploration budget split."""
    X, y, t = stream["X"], stream["y"], stream["t"]
    w_post = stream["w_post"]
    rng = np.random.default_rng(seed)

    pre = t < T_SHUTDOWN
    base = fit(kind, X[pre], y[pre], seed)

    adaptive = None
    comp = {"base": 0.5, "adp": 0.5}
    revealed = {}
    caught_post, pr_post, angles = 0, [], []

    for s in range(T_SHUTDOWN, SYNTH_T):
        m = t == s
        Xs, ys = X[m], y[m]

        p_base = proba(base, Xs)
        if adaptive is not None:
            p_adp = proba(adaptive, Xs)
            logits = np.array([comp["base"], comp["adp"]]) / COMP_TAU
            logits -= logits.max()
            w = np.exp(logits); w /= w.sum()
            p = w[0]*p_base + w[1]*p_adp
        else:
            p_adp, p = None, p_base

        kk = min(k, len(Xs))
        top = np.argsort(p)[::-1][:kk]
        caught_post += int(np.sum(ys[top] == 1))
        pr_post.append(safe_ap(ys, p))

        # geometry: angle of CURRENT effective boundary to true post-concept
        if kind == "logistic":
            eff = adaptive if adaptive is not None else base
            angles.append(concept_angle(eff.coef_.ravel(), w_post))

        if explore:
            n_x = int(EPSILON * kk)
            exploit = top[:kk - n_x]
            rest = np.setdiff1d(np.arange(len(Xs)), exploit)
            ex = rng.choice(rest, size=min(n_x, len(rest)), replace=False)
            chosen = np.concatenate([exploit, ex])
        else:
            chosen = top
        gi = np.where(m)[0][chosen]
        revealed[s] = gi

        if adaptive is not None and (ys[chosen] == 1).any() and (ys[chosen] == 0).any():
            for nm, pp in [("base", p_base), ("adp", p_adp)]:
                comp[nm] = COMP_EMA*comp[nm] + (1-COMP_EMA)*safe_ap(ys[chosen], pp[chosen])

        avail = [q for q in revealed if s - LABEL_DELAY - ADAPT_WINDOW < q <= s - LABEL_DELAY]
        if avail:
            pool = np.concatenate([revealed[q] for q in avail])
            yp = y[pool]
            if (yp == 1).sum() >= MIN_POS and (yp == 0).sum() >= MIN_NEG:
                adaptive = fit(kind, X[pool], yp, seed)

    n_illicit_post = int((y[t >= T_SHUTDOWN] == 1).sum())
    return {"caught_post": caught_post, "n_illicit_post": n_illicit_post,
            "mean_pr_auc_post": float(np.nanmean(pr_post)),
            "final_angle": float(angles[-1]) if angles else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    mdir = Path(cfg["paths"]["results_metrics"])

    rows, done = [], 0
    total = len(THETAS)*len(RATES)*len(BUDGETS)*len(args.seeds)*2
    for theta in THETAS:
        for rate in RATES:
            for seed in args.seeds:
                set_seed(seed)
                stream = generate_stream(theta, rate, seed)
                for k in BUDGETS:
                    for kind, explore in [("lgbm", True), ("lgbm", False)]:
                        r = run(kind, explore, k, stream, seed)
                        rows.append({"detector": kind, "explore": explore,
                                     "theta": round(theta, 4), "base_rate": rate,
                                     "k": k, "seed": seed, **r})
                        done += 1
                # logistic geometry track: greedy only, k=50
                rg = run("logistic", False, 50, stream, seed)
                rows.append({"detector": "logistic", "explore": False,
                             "theta": round(theta, 4), "base_rate": rate,
                             "k": 50, "seed": seed, **rg})
                done += 1
                print(f"[{done:4d}/~{total}] theta={theta:.3f} p={rate} "
                      f"seed={seed} done")

    res = pd.DataFrame(rows)
    res.to_csv(mdir / "synth_phase_results.csv", index=False)

    # ---- Prediction 1: collapse vs theta (static = greedy, no adaptation window hit yet at t*) ----
    print("\n=== P1: LightGBM greedy, mean caught/illicit vs theta (k=50) ===")
    sub = res[(res.detector=="lgbm") & (~res.explore) & (res.k==50)]
    print(sub.groupby("theta").apply(
        lambda g: pd.Series({
            "caught_mean": g.caught_post.mean(),
            "illicit_mean": g.n_illicit_post.mean(),
            "catch_frac": g.caught_post.sum()/max(g.n_illicit_post.sum(),1)})
    ).round(3).to_string())

    print("\n=== P2/P4: caught vs (theta, k), greedy vs explore (LightGBM, p=0.025) ===")
    sub = res[(res.detector=="lgbm") & (res.base_rate==0.025)]
    piv = sub.groupby(["explore","theta","k"]).caught_post.mean().round(1)
    print(piv.to_string())

    print("\n=== Mechanism: logistic boundary angle to true post-concept (rad) vs theta ===")
    sub = res[res.detector=="logistic"]
    print(sub.groupby("theta").agg(
        final_angle_mean=("final_angle","mean"),
        caught_mean=("caught_post","mean")).round(3).to_string())

    print(f"\nSaved -> {mdir/'synth_phase_results.csv'}")


if __name__ == "__main__":
    main()