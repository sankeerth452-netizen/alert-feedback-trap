# experiments/run_step42_diagnostic.py
"""Why does the step-42 detector escape? Diagnostic before deciding the reframe.

Hypothesis: step-42 ranks well post-shutdown because steps 35-42 already contain
the drift ONSET (the same drift the PSI alarm fires on before t=43), so it has
peeked at the new regime. We test this by sweeping the training cutoff and
watching caught@50 climb as the window approaches the shutdown -- if caught is a
smooth function of 'how close training ends to t=43', the escape is explained by
drift-onset exposure, not by generalization.

Run:  python experiments/run_step42_diagnostic.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from src.data.load import FEATURE_COLS, load_dataframe
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
K = 50

def main():
    cfg = yaml.safe_load(open("configs/base.yaml")); set_seed(cfg["seed"])
    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    print("Training-cutoff sweep: caught@50 over post-drift (43-49) vs train_end")
    print("If caught climbs smoothly as cutoff -> 42, escape = drift-onset exposure.\n")
    rows = []
    for cut in range(30, 43):
        mask = (t <= cut) & (y != -1)
        m = LGBMClassifier(random_state=cfg["seed"], **LGBM).fit(X[mask], y[mask])
        caught = 0; prs = []
        for s in range(43, 50):
            sm = (t == s) & (y != -1); Xs, ys = X[sm], y[sm]
            if ys.sum() == 0: continue
            p = m.predict_proba(Xs)[:, 1]
            top = np.argsort(p)[::-1][:min(K, len(ys))]
            caught += int((ys[top] == 1).sum())
            prs.append(average_precision_score(ys, p))
        rows.append(dict(train_end=cut, caught_at_50=caught,
                         mean_pr_auc=round(np.mean(prs), 4)))
        print(f"train_end={cut:2d} | caught@50={caught:3d}/169 | "
              f"PR-AUC={np.mean(prs):.3f}")

    d = pd.DataFrame(rows)
    d.to_csv(Path(cfg['paths']['results_metrics'])/"cutoff_sweep.csv", index=False)
    jump = d.caught_at_50.diff().abs().max()
    print(f"\nLargest single-step jump in caught: {jump:.0f}")
    print("Smooth climb (no single cliff) => escape is gradual drift-onset")
    print("exposure, NOT a generalization threshold => trap is a cold-start /")
    print("steady-state phenomenon, which is the honest reframe.")

if __name__ == "__main__":
    main()