"""Sanity check: does the v2 synthetic environment actually produce a CLIFF?

Two mandatory checks at one seed, no adaptation:
  (A) static post-drift catch-fraction at theta=pi/2 must fall near/below the
      random-investigation rate (k / n_per_step).
  (B) a pre-drift-trained logistic boundary must sit near pi/2 (orthogonal) to
      the true post-concept at theta=pi/2, and near 0 at theta=0.
If these fail, the generator is still wrong -- do NOT run the full sweep.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.synth.generator import (generate_stream, concept_angle,
                                  T_SHUTDOWN, T, N_PER_STEP)
from src.utils.seed import set_seed

LGBM = dict(n_estimators=300, num_leaves=48, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
K = 50
RATE = 0.025


def static_eval(theta, seed=42):
    set_seed(seed)
    s = generate_stream(theta, RATE, seed)
    X, y, t = s["X"], s["y"], s["t"]
    pre = t < T_SHUTDOWN
    gb = LGBMClassifier(random_state=seed, **LGBM).fit(X[pre], y[pre])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=seed).fit(X[pre], y[pre])

    caught = illicit = 0
    prs = []
    for step in range(T_SHUTDOWN, T):
        m = t == step
        Xs, ys = X[m], y[m]
        p = gb.predict_proba(Xs)[:, 1]
        kk = min(K, len(Xs))
        top = np.argsort(p)[::-1][:kk]
        caught += int((ys[top] == 1).sum())
        illicit += int((ys == 1).sum())
        if (ys == 1).any() and (ys == 0).any():
            prs.append(average_precision_score(ys, p))

    angle = concept_angle(lr.coef_.ravel(), s["w_post"])
    rand_rate = K / N_PER_STEP
    return {"theta": round(theta, 3),
            "catch_frac": round(caught / max(illicit, 1), 3),
            "illicit_post": illicit,
            "random_rate": round(rand_rate, 3),
            "pr_auc": round(float(np.mean(prs)), 3),
            "boundary_angle_rad": round(angle, 3),
            "boundary_angle_deg": round(np.degrees(angle), 1)}


def main():
    print(f"{'theta':>6} {'catch_frac':>11} {'rand_rate':>10} "
          f"{'pr_auc':>7} {'angle_deg':>10} {'illicit':>8}")
    rows = [static_eval(0.0), static_eval(np.pi/2)]
    for r in rows:
        print(f"{r['theta']:>6} {r['catch_frac']:>11} {r['random_rate']:>10} "
              f"{r['pr_auc']:>7} {r['boundary_angle_deg']:>10} {r['illicit_post']:>8}")

    t0, t90 = rows
    print("\n--- CLIFF CHECKS ---")
    okA = t90["catch_frac"] <= 2 * t90["random_rate"]
    okB = t90["boundary_angle_deg"] > 60 and t0["boundary_angle_deg"] < 30
    print(f"(A) theta=pi/2 catch_frac {t90['catch_frac']} <= ~2x random "
          f"{2*t90['random_rate']:.3f}? {'PASS' if okA else 'FAIL'}")
    print(f"(B) angle: theta=0 -> {t0['boundary_angle_deg']} deg (want <30), "
          f"theta=pi/2 -> {t90['boundary_angle_deg']} deg (want >60)? "
          f"{'PASS' if okB else 'FAIL'}")
    print(f"\n{'BOTH PASS -> run the full sweep.' if okA and okB else 'FAIL -> generator still needs work; do not sweep.'}")


if __name__ == "__main__":
    main()