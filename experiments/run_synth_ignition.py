# experiments/run_synth_ignition.py
"""Run the ALERT-ONLY feedback loop on the synthetic rotating-concept stream,
to demonstrate the TRAP and bimodal IGNITION in a second environment (not just
collapse). This closes the biggest external-validity gap.

Fix theta=pi/2 (orthogonal drift = full collapse, matching Elliptic), sweep
budget k near the critical region, greedy vs eps10, many seeds. Records per-run
caught so we can show the bimodal trapped-or-ignited split.

Run:  python experiments/run_synth_ignition.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.synth.generator import generate_stream, T_SHUTDOWN, T as SYNTH_T
from src.utils.seed import set_seed

LGBM = dict(n_estimators=300, num_leaves=48, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
THETA = np.pi / 2          # orthogonal: full collapse to random (matches Elliptic)
RATE = 0.025
MIN_POS, MIN_NEG, ADAPT_WINDOW, DELAY = 5, 5, 6, 1
COMP_TAU, COMP_EMA = 0.10, 0.6
BUDGETS = [25, 50, 75, 100, 125, 150]


def safe_ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if (y==1).any() and (y==0).any() else np.nan


def run(explore_eps, k, stream, seed):
    X, y, t = stream["X"], stream["y"], stream["t"]
    rng = np.random.default_rng(seed)
    pre = t < T_SHUTDOWN
    base = LGBMClassifier(random_state=seed, **LGBM).fit(X[pre], y[pre])
    adaptive, comp, revealed = None, {"base":0.5,"adp":0.5}, {}
    caught = 0
    for s in range(T_SHUTDOWN, SYNTH_T):
        m = t == s; Xs, ys = X[m], y[m]
        p_base = base.predict_proba(Xs)[:,1]
        if adaptive is not None:
            p_adp = adaptive.predict_proba(Xs)[:,1]
            lg = np.array([comp["base"], comp["adp"]])/COMP_TAU; lg -= lg.max()
            w = np.exp(lg); w /= w.sum(); p = w[0]*p_base + w[1]*p_adp
        else:
            p_adp, p = None, p_base
        kk = min(k, len(Xs))
        top = np.argsort(p)[::-1][:kk]
        caught += int((ys[top]==1).sum())
        # investigation with eps-exploration
        n_x = int(explore_eps*kk)
        exploit = top[:kk-n_x]
        rest = np.setdiff1d(np.arange(len(Xs)), exploit)
        ex = rng.choice(rest, size=min(n_x,len(rest)), replace=False) if n_x>0 else np.array([],int)
        chosen = np.concatenate([exploit, ex]).astype(int)
        revealed[s] = (np.where(m)[0][chosen], ys[chosen])
        if adaptive is not None and (ys[chosen]==1).any() and (ys[chosen]==0).any():
            for nm,pp in [("base",p_base),("adp",p_adp)]:
                comp[nm] = COMP_EMA*comp[nm] + (1-COMP_EMA)*safe_ap(ys[chosen], pp[chosen])
        avail = [q for q in revealed if s-DELAY-ADAPT_WINDOW < q <= s-DELAY]
        if avail:
            pi = np.concatenate([revealed[q][0] for q in avail])
            pl = np.concatenate([revealed[q][1] for q in avail])
            if (pl==1).sum()>=MIN_POS and (pl==0).sum()>=MIN_NEG:
                adaptive = LGBMClassifier(random_state=seed, **LGBM).fit(X[pi], pl)
    return caught


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=list(range(42, 52)))
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/base.yaml"))
    mdir = Path(cfg["paths"]["results_metrics"]); fdir = Path(cfg["paths"]["results_figures"])
    mdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        stream = generate_stream(THETA, RATE, seed)
        n_ill_post = int((stream["y"][stream["t"]>=T_SHUTDOWN]==1).sum())
        for k in BUDGETS:
            for eps, name in [(0.0,"greedy"), (0.1,"eps10")]:
                c = run(eps, k, stream, seed)
                rows.append({"policy":name,"k":k,"seed":seed,
                             "caught":c,"n_illicit_post":n_ill_post})
        print(f"seed={seed} done (n_illicit_post~{n_ill_post})")

    res = pd.DataFrame(rows)
    res.to_csv(mdir / "synth_ignition.csv", index=False)

    # ignition fraction: a run 'ignites' if caught exceeds 2x the greedy-at-low-k floor
    floor = res[(res.policy=="greedy") & (res.k==25)]["caught"].mean()
    thresh = max(2*floor, floor+10)
    print(f"\nIgnition threshold (caught >): {thresh:.0f}")
    agg = res.groupby(["policy","k"]).agg(
        caught_mean=("caught","mean"), caught_std=("caught","std"),
        p_ignite=("caught", lambda x: float((x>thresh).mean()))).round(2)
    print(agg.to_string())

    # Figure: bimodal scatter on synthetic (mirrors Elliptic fig_ignition)
    fig, ax = plt.subplots(figsize=(3.4,2.6))
    for name,c in [("eps10","#2E7D5B"),("greedy","#16324F")]:
        d = res[res.policy==name]
        ax.scatter(d["k"]+ (1.5 if name=="greedy" else -1.5), d["caught"],
                   s=18, alpha=0.7, label=name,
                   marker="s" if name=="greedy" else "o",
                   color=c, edgecolor="white", linewidth=0.4)
    ax.axhline(thresh, color="#8FA0B4", ls=":", lw=1.0)
    ax.set_xlabel("Investigation budget $k$ / step")
    ax.set_ylabel("Illicit caught (synthetic)")
    ax.set_title("Bimodal ignition reproduces on synthetic drift")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(fdir / f"fig_synth_ignition.{ext}", dpi=300)
    print(f"\nSaved fig_synth_ignition.pdf -> {fdir}")
    print("If the eps10 points split into a trapped cluster and an ignited "
          "cluster near critical k, the trap is demonstrated in a SECOND "
          "environment -- add this panel to the synthetic figure and update Sec 9.")


if __name__ == "__main__":
    main()