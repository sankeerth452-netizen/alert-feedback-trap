# experiments/run_step42_canonical.py
"""Definitive realistic-base numbers via the CANONICAL competence-gate path,
to replace the PSI-gated numbers that leaked in from confirm.py.

Reuses run_aegis_final.run_mode with mode='aegis_cg' (competence gate), but with
the base trained through step 42. We monkeypatch TRAIN_END only for the base by
calling run_mode with a train_end of 42 so the base sees all pre-shutdown data.

Reports: frozen (static), bulk, and aegis_cg alert-only at eps=0/0.1/0.2.
30 seeds for stochastic arms.

Run:  caffeinate -i python experiments/run_step42_canonical.py --n-seeds 30
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, yaml
import experiments.run_aegis_final as RAF
from experiments.run_aegis_final import run_mode
from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed

def caught_post(rows):
    return int(sum(r["illicit_caught"] for r in rows if r["time_step"] >= 43))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args(); seeds = list(range(42, 42+args.n_seeds))
    cfg = yaml.safe_load(open("configs/base.yaml"))
    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    TRAIN_END = 42                      # realistic base
    ref = X[t <= TRAIN_END]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, 35)])
    psi_z = {s: (psi[s]-tv.mean())/tv.std(ddof=1) for s in psi}

    set_seed(seeds[0])
    static = caught_post(run_mode("static", X, y, t, TRAIN_END, psi_z, seeds[0]))
    bulk   = caught_post(run_mode("adaptive_full", X, y, t, TRAIN_END, psi_z, seeds[0]))
    print(f"\n=== STEP-42 BASE via CANONICAL run_aegis_final ===")
    print(f"frozen (static)      : {static} /169")
    print(f"bulk (adaptive_full) : {bulk} /169")

    for eps in [0.0, 0.1, 0.2]:
        RAF.EPSILON = eps               # canonical aegis_cg uses module-level EPSILON
        vals = np.array([caught_post(run_mode("aegis_cg", X, y, t, TRAIN_END, psi_z, sd))
                         for sd in seeds])
        tag = "greedy" if eps == 0 else f"eps={eps}"
        print(f"aegis_cg {tag:8s}   : {vals.mean():.1f} +/- {vals.std(ddof=1):.1f}")
    print("\nThese are the canonical competence-gated numbers for the realistic")
    print("base -> they go straight into Sec 6, replacing the PSI-gated 81/87.9/93.5.")

if __name__ == "__main__":
    main()