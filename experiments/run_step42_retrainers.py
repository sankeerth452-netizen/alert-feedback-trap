# experiments/run_step42_retrainers.py
"""Reviewer point #1: is the 113->81 harm specific to AEGIS's competence gate,
or does ANY retrainer on the realistic base inherit it? Run alternative
alert-only retraining schemes on the frozen step-42 base and report caught@50.

Schemes (all alert-only, k=50, eps=0 greedy unless noted):
  - frozen            : no retraining (anchor, =113)
  - replace           : adaptive expert REPLACES base once trained (no gate)
  - average           : 0.5*base + 0.5*expert (no competence gate)
  - aegis_cg          : competence-gated blend (=81.0, the paper's number)
30 seeds for stochastic schemes.

Run:  caffeinate -i python experiments/run_step42_retrainers.py --n-seeds 30
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

def run(scheme, eps, X, y, t, psi_z, seed):
    rng = np.random.default_rng(seed)
    base = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= TRAIN_END_BASE) & (y != -1)
    base.fit(X[bm], y[bm]); base_ref = X[t <= TRAIN_END_BASE]
    adaptive, adaptive_ref, comp, revealed, caught = None, None, {"base":0.5,"adp":0.5}, {}, 0
    for s in range(TRAIN_END_BASE + 1, 50):
        sm = np.where((t == s) & (y != -1))[0]; Xs, ys = X[sm], y[sm]
        if len(ys) == 0: continue
        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None and scheme != "frozen":
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            if scheme == "replace":
                prob = p_adp
            elif scheme == "average":
                prob = 0.5*p_base + 0.5*p_adp
            elif scheme == "aegis_cg":
                lg = np.array([comp["base"], comp["adp"]])/COMP_TAU; lg -= lg.max()
                w = np.exp(lg); w /= w.sum(); prob = w[0]*p_base + w[1]*p_adp
        else:
            p_adp, prob = None, p_base
        kk = min(BUDGET_K, len(ys)); top = np.argsort(prob)[::-1][:kk]
        caught += int((ys[top] == 1).sum())
        if scheme == "frozen": continue
        n_x = int(eps*kk); exploit = top[:kk-n_x]
        rest = np.setdiff1d(np.arange(len(ys)), exploit)
        ex = rng.choice(rest, size=min(n_x,len(rest)), replace=False) if n_x>0 else np.array([],int)
        chosen = np.concatenate([exploit, ex]).astype(int)
        revealed[s] = sm[chosen]
        if scheme == "aegis_cg" and adaptive is not None:
            yl = ys[chosen]
            if (yl==1).any() and (yl==0).any():
                comp["base"]=COMP_EMA*comp["base"]+(1-COMP_EMA)*safe_ap(yl,p_base[chosen])
                comp["adp"] =COMP_EMA*comp["adp"] +(1-COMP_EMA)*safe_ap(yl,p_adp[chosen])
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
    print(f"\n=== STEP-42 BASE, alert-only retraining schemes (caught@50) ===")
    print(f"{'frozen (no retrain)':28s}: {run('frozen',0,X,y,t,psi_z,seeds[0])} /169")
    for scheme in ["replace","average","aegis_cg"]:
        vals = np.array([run(scheme,0.0,X,y,t,psi_z,sd) for sd in seeds])
        print(f"{scheme+' (greedy)':28s}: {vals.mean():.1f} +/- {vals.std(ddof=1):.1f}")
    print("\nIf replace AND average also fall well below 113, the harm is the")
    print("FEEDBACK LOOP, not AEGIS's gate -> 'the loop, not the model' holds")
    print("for the realistic base too. Reports go straight into Sec 6.")

if __name__ == "__main__":
    main()