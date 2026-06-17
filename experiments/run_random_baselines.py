# experiments/run_random_baseline.py
"""Exact per-step random-investigation baseline, to ground the claim that the
static detector's 14/169 caught is 'no better than a small multiple of random'.

For each post-drift step we investigate k of the n labeled transactions; the
number of illicit caught is Hypergeometric(n, n_illicit, k). We report the
summed expectation across steps (the right 'random' reference) and a Monte-Carlo
check, and contrast with the static detector's 14.

Run:  python experiments/run_random_baseline.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from src.data.load import FEATURE_COLS, load_dataframe

K = 50
N_MC = 100_000


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    df = load_dataframe(cfg["paths"]["raw_dir"])
    y, t = df["label"].values, df["time_step"].values
    rng = np.random.default_rng(0)

    rows, exp_total, mc_caught = [], 0.0, np.zeros(N_MC)
    for s in range(43, 50):
        m = (t == s) & (y != -1)
        ys = y[m]
        n, n_ill = len(ys), int((ys == 1).sum())
        if n == 0:
            continue
        k = min(K, n)
        exp_step = k * n_ill / n                      # hypergeometric mean
        exp_total += exp_step
        # Monte-Carlo: draw k of n, count illicit, vectorized across trials
        for i in range(0, N_MC, 20000):               # chunk to save memory
            j = min(i + 20000, N_MC)
            draws = np.array([rng.permutation(n)[:k] for _ in range(j - i)])
            mc_caught[i:j] = (ys[draws] == 1).sum(axis=1)
        rows.append({"step": s, "n": n, "n_illicit": n_ill,
                     "k": k, "exp_caught": round(exp_step, 3)})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    mc_mean = mc_caught.reshape(-1)[:N_MC]
    # NOTE: MC above re-randomizes per step pool; for an honest total we redo MC jointly:
    # joint Monte-Carlo over all steps
    joint = np.zeros(N_MC)
    pools = [(int(((t == s) & (y != -1)).sum()),
              int(((t == s) & (y == 1)).sum())) for s in range(43, 50)
             if ((t == s) & (y != -1)).sum() > 0]
    ys_by_step = [y[(t == s) & (y != -1)] for s in range(43, 50)
                  if ((t == s) & (y != -1)).sum() > 0]
    for trial in range(N_MC):
        c = 0
        for ys_s in ys_by_step:
            n = len(ys_s); k = min(K, n)
            idx = rng.permutation(n)[:k]
            c += int((ys_s[idx] == 1).sum())
        joint[trial] = c
        if trial >= 20000:            # 20k trials is plenty; truncate for speed
            joint = joint[:trial + 1]
            break

    print(f"\nSum of per-step hypergeometric expectation : {exp_total:.2f}")
    print(f"Joint Monte-Carlo mean caught (random)     : {joint.mean():.2f} "
          f"+/- {joint.std():.2f}")
    print(f"Monte-Carlo 95th percentile                : "
          f"{np.percentile(joint, 95):.0f}")
    print(f"Static detector caught                     : 14")
    p_ge_14 = (joint >= 14).mean()
    print(f"P(random >= 14)                            : {p_ge_14:.3f}")
    print("\nInterpretation: if E[random] is ~9 and P(random>=14) is non-trivial,"
          "\nthe static detector is NOT distinguishable from random at k=50 -- "
          "report E[random] explicitly and phrase as 'no better than random'.")
    Path(cfg["paths"]["results_metrics"]).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(cfg["paths"]["results_metrics"]) / "random_baseline.csv",
               index=False)


if __name__ == "__main__":
    main()