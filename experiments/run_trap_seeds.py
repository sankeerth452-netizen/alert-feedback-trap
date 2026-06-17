# experiments/run_trap_seeds.py
"""30-seed alert-only trap experiment with Welch t-test, to make the
'methods become statistically indistinguishable' claim robust (reviewer asked
for 20-30 seeds; p~0.12 on 10 seeds is borderline).

Wraps the EXISTING simulators:
  - run_stream_baselines.run(model, regime, ...)  with regime="budget"
  - run_aegis_final.run_mode(mode, ...)           with mode="aegis_cg" / "static"
Both return lists of per-step dicts with 'illicit_caught'; we sum post-drift.

Run:  caffeinate -i python experiments/run_trap_seeds.py --n-seeds 30
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
from scipy import stats

from src.data.load import FEATURE_COLS, load_dataframe
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed
from experiments.run_stream_baselines import run as run_stream
from experiments.run_aegis_final import run_mode as run_aegis


def caught_post(rows):
    """Sum illicit_caught over post-drift steps (t>=43) from a list of row dicts."""
    return int(sum(r["illicit_caught"] for r in rows if r["time_step"] >= 43))


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return (x.mean() - h, x.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-start", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()
    seeds = list(range(args.seeds_start, args.seeds_start + args.n_seeds))

    cfg = yaml.safe_load(open("configs/base.yaml"))
    train_end = cfg["split"]["train_end"]
    mdir = Path(cfg["paths"]["results_metrics"]); mdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    # AEGIS needs the calibrated PSI alarm precomputed (same as its main()).
    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, train_end + 1)])
    psi_z = {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}

    rows = []
    # Static is deterministic -> compute once via aegis run_mode("static").
    set_seed(seeds[0])
    static_caught = caught_post(run_aegis("static", X, y, t, train_end, psi_z, seeds[0]))
    rows.append({"method": "Static", "seed": seeds[0], "caught": static_caught})
    print(f"Static (deterministic) caught={static_caught}")

    for sd in seeds:
        set_seed(sd)
        arf = caught_post(run_stream("arf", "budget", X, y, t, train_end, sd))
        rows.append({"method": "ARF", "seed": sd, "caught": arf})
        set_seed(sd)
        srp = caught_post(run_stream("srp", "budget", X, y, t, train_end, sd))
        rows.append({"method": "SRP", "seed": sd, "caught": srp})
        set_seed(sd)
        cg = caught_post(run_aegis("aegis_cg", X, y, t, train_end, psi_z, sd))
        rows.append({"method": "AEGIS-CG", "seed": sd, "caught": cg})
        print(f"seed={sd:3d} | ARF={arf:3d}  SRP={srp:3d}  AEGIS-CG={cg:3d}")

    res = pd.DataFrame(rows)
    res.to_csv(mdir / "trap_seeds.csv", index=False)

    print(f"\n===== ALERT-ONLY (budget), k post-drift, {args.n_seeds} seeds =====")
    for name in ["Static", "ARF", "SRP", "AEGIS-CG"]:
        x = res[res.method == name]["caught"].values
        lo, hi = ci95(x)
        sd_ = x.std(ddof=1) if len(x) > 1 else 0.0
        print(f"{name:9s} mean={x.mean():6.2f}  std={sd_:5.2f}"
              f"  95%CI=[{lo:5.2f},{hi:5.2f}]  n={len(x)}")

    arf = res[res.method == "ARF"]["caught"].values
    cg  = res[res.method == "AEGIS-CG"]["caught"].values
    tstat, p = stats.ttest_ind(arf, cg, equal_var=False)
    print(f"\nWelch ARF vs AEGIS-CG:  t={tstat:.3f}  p={p:.4f}  "
          f"(n_arf={len(arf)}, n_cg={len(cg)})")
    if p < 0.05:
        print(">>> SIGNIFICANT at 30 seeds. Reframe Sec 6/Table 3: method choice "
              "is NOT second-order; ARF beats AEGIS, yet BOTH are dominated by "
              "the feedback regime (bulk 53/98 -> alert-only ~teens/20s).")
    else:
        print(">>> NOT significant: 'differences vanish under alert-only "
              "feedback' holds at 30 seeds. Update Table 3 numbers + p-value.")


if __name__ == "__main__":
    main()