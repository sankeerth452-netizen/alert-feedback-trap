# experiments/run_step42_confirm.py
"""Confirm the step-42 trap: separate alert-only HARM from exploration COST.

For the step-42 base, run alert-only at several exploration levels including
greedy (eps=0). If greedy alert-only is already << static (113), the trap binds
on feedback alone, independent of exploration. Also reports bulk and static as
anchors. 30 seeds for stochastic arms.

Run:  caffeinate -i python experiments/run_step42_confirm.py --n-seeds 30
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
BUDGET_K, ALARM_Z = 50, 2.0
MIN_POS, MIN_NEG, ADAPT_WINDOW, DELAY = 5, 5, 8, 1
COMP_TAU, COMP_EMA = 0.10, 0.6
TRAIN_END_BASE = 42

def safe_ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if (y==1).any() and (y==0).any() else np.nan

def run(regime, eps, X, y, t, psi_z, seed):
    rng = np.random.default_rng(seed)
    base = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= TRAIN_END_BASE) & (y != -1)
    base.fit(X[bm], y[bm]); base_ref = X[t <= TRAIN_END_BASE]
    adaptive, adaptive_ref, revealed, caught = None, None, {}, 0
    for s in range(TRAIN_END_BASE + 1, 50):
        sm = np.where((t == s) & (y != -1))[0]; Xs, ys = X[sm], y[sm]
        if len(ys) == 0: continue
        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None and regime != "static":
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            psis = np.array([mean_psi(base_ref, Xs), mean_psi(adaptive_ref, Xs)])
            lg = -psis/0.05; lg -= lg.max(); w = np.exp(lg); w /= w.sum()
            prob = w[0]*p_base + w[1]*p_adp
        else:
            prob = p_base
        kk = min(BUDGET_K, len(ys)); top = np.argsort(prob)[::-1][:kk]
        caught += int((ys[top] == 1).sum())
        if regime == "static": continue
        if regime == "bulk":
            chosen = np.arange(len(ys))
        else:
            n_x = int(eps*kk); exploit = top[:kk-n_x]
            rest = np.setdiff1d(np.arange(len(ys)), exploit)
            ex = rng.choice(rest, size=min(n_x,len(rest)), replace=False) if n_x>0 else np.array([],int)
            chosen = np.concatenate([exploit, ex]).astype(int)
        revealed[s] = sm[chosen]
        if psi_z[s] >= ALARM_Z or adaptive is not None:
            avail = [q for q in revealed if s-DELAY-ADAPT_WINDOW < q <= s-DELAY]
            if avail:
                pool = np.concatenate([revealed[q] for q in avail]); yp = y[pool]
                if (yp==1).sum()>=MIN_POS and (yp==0).sum()>=MIN_NEG:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM).fit(X[pool], yp)
                    adaptive_ref = X[np.isin(t, avail)]
    return caught

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args(); seeds = list(range(42, 42+args.n_seeds))
    cfg = yaml.safe_load(open("configs/base.yaml"))
    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values
    ref = X[t <= TRAIN_END_BASE]
    psi = {s: mean_psi(ref, X[t==s]) for s in range(1,50)}
    tv = np.array([psi[s] for s in range(1,35)])
    psi_z = {s: (psi[s]-tv.mean())/tv.std(ddof=1) for s in psi}

    set_seed(seeds[0])
    print(f"\n===== STEP-42 BASE confirmation (caught@50 over 43-49) =====")
    print(f"{'Static (no adaptation)':30s}: {run('static',0,X,y,t,psi_z,seeds[0])} /169")
    print(f"{'Bulk feedback':30s}: {run('bulk',0,X,y,t,psi_z,seeds[0])} /169")
    for eps in [0.0, 0.1, 0.2]:
        vals = np.array([run('alert_only', eps, X, y, t, psi_z, sd) for sd in seeds])
        print(f"{'Alert-only eps='+str(eps):30s}: {vals.mean():.1f} +/- {vals.std(ddof=1):.1f}")
    print("\nIf greedy (eps=0.0) alert-only is already well below 113 -> trap binds")
    print("on FEEDBACK alone (not exploration). That is the clean, strong result.")

if __name__ == "__main__":
    main()