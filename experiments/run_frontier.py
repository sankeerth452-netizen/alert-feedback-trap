"""The label-acquisition frontier: budget x exploration policy sweep.

Fixed adaptation machinery (base LGBM + alarm-spawned adaptive expert +
competence gate). What varies is HOW the investigation budget is spent:

  greedy      : all k on top-scored alerts (pure exploitation)
  eps10/20/40 : (1-eps)*k exploit + eps*k uniform-random explore
  stratified  : 0.8*k exploit + 0.2*k spread uniformly across score deciles
  novelty     : 0.8*k exploit + 0.2*k on most train-atypical samples
                (mean |z| of features vs train distribution)

Results APPEND to frontier_results.csv across invocations, so zoom runs
(e.g. --policies greedy --budgets 55 60 ... 95) merge with the main sweep.
Per-step records go to frontier_step_records.csv for the mediation analysis
(model quality at step t vs positives acquired BEFORE t).

Runs:
  python experiments/run_frontier.py --seeds 42 43 44 45 46
  python experiments/run_frontier.py --policies greedy --budgets 55 60 65 70 75 80 85 90 95 --seeds 42
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
LABEL_DELAY, ALARM_Z = 1, 2.0
MIN_POS, MIN_NEG, ADAPT_WINDOW = 5, 5, 8
COMP_TAU, COMP_EMA = 0.10, 0.6
BUDGETS = [25, 50, 100, 200]
POLICIES = ["greedy", "eps10", "eps20", "eps40", "stratified", "novelty"]


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def choose(policy, prob, novelty_scores, k, rng):
    """Return indices (into the step) to investigate under a policy."""
    order = np.argsort(prob)[::-1]
    if policy == "greedy":
        return order[:k]
    eps = {"eps10": 0.1, "eps20": 0.2, "eps40": 0.4,
           "stratified": 0.2, "novelty": 0.2}[policy]
    n_x = max(1, int(eps * k))
    exploit = order[: k - n_x]
    rest = np.setdiff1d(np.arange(len(prob)), exploit)
    if len(rest) == 0:
        return exploit
    if policy in ("eps10", "eps20", "eps40"):
        explore = rng.choice(rest, size=min(n_x, len(rest)), replace=False)
    elif policy == "stratified":
        deciles = np.array_split(rest[np.argsort(prob[rest])[::-1]], 10)
        picks, i = [], 0
        while len(picks) < min(n_x, len(rest)):
            d = deciles[i % 10]
            if len(d):
                picks.append(rng.choice(d))
                deciles[i % 10] = d[d != picks[-1]]
            i += 1
        explore = np.array(picks)
    else:  # novelty
        explore = rest[np.argsort(novelty_scores[rest])[::-1][:n_x]]
    return np.concatenate([exploit, explore])


def run_sim(policy, k, X, y, t, train_end, psi_z, novelty_all, seed):
    rng = np.random.default_rng(seed)
    base = LGBMClassifier(random_state=seed, **LGBM)
    bm = (t <= train_end) & (y != -1)
    base.fit(X[bm], y[bm])

    adaptive = None
    comp = {"base": 0.5, "adp": 0.5}
    revealed = {}
    caught_post, pr_post, pos_acquired = 0, [], 0
    step_rows = []

    for s in range(train_end + 1, 50):
        idx_s = np.where((t == s) & (y != -1))[0]
        Xs, ys = X[idx_s], y[idx_s]

        p_base = base.predict_proba(Xs)[:, 1]
        if adaptive is not None:
            p_adp = adaptive.predict_proba(Xs)[:, 1]
            logits = np.array([comp["base"], comp["adp"]]) / COMP_TAU
            logits -= logits.max()
            w = np.exp(logits); w /= w.sum()
            prob = w[0] * p_base + w[1] * p_adp
        else:
            p_adp, prob = None, p_base

        kk = min(k, len(idx_s))
        chosen = choose(policy, prob, novelty_all[idx_s], kk, rng)

        # ---- mediation record: quality NOW vs positives acquired BEFORE ----
        # (with LABEL_DELAY=1, pos_acquired at this point is exactly what the
        #  current adaptive expert could have been trained on)
        if s >= 43:
            step_rows.append({"policy": policy, "k": k, "seed": seed,
                              "time_step": s,
                              "cum_pos_before": pos_acquired,
                              "pr_auc": safe_ap(ys, prob),
                              "caught_step": int(np.sum(ys[chosen] == 1))})
            caught_post += int(np.sum(ys[chosen] == 1))
            pr_post.append(safe_ap(ys, prob))

        revealed[s] = idx_s[chosen]
        pos_acquired += int(np.sum(ys[chosen] == 1)) if s >= 43 else 0

        yl = ys[chosen]
        if adaptive is not None and (yl == 1).any() and (yl == 0).any():
            for name, p in [("base", p_base), ("adp", p_adp)]:
                comp[name] = COMP_EMA * comp[name] + \
                             (1 - COMP_EMA) * safe_ap(yl, p[chosen])

        if psi_z[s] >= ALARM_Z or adaptive is not None:
            avail = [q for q in revealed
                     if s - LABEL_DELAY - ADAPT_WINDOW < q <= s - LABEL_DELAY]
            if avail:
                pool = np.concatenate([revealed[q] for q in avail])
                yp = y[pool]
                if (yp == 1).sum() >= MIN_POS and (yp == 0).sum() >= MIN_NEG:
                    adaptive = LGBMClassifier(random_state=seed, **LGBM)
                    adaptive.fit(X[pool], yp)

    summary = {"policy": policy, "k": k, "seed": seed,
               "caught_post": caught_post,
               "pos_labels_acquired_post": pos_acquired,
               "mean_pr_auc_post": float(np.nanmean(pr_post))}
    return summary, step_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--budgets", nargs="*", type=int, default=BUDGETS)
    parser.add_argument("--policies", nargs="*", default=POLICIES)
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

    # novelty score: mean |z| of features vs train distribution (label-free)
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-8
    novelty_all = np.abs((X - mu) / sd).mean(axis=1)

    rows, step_records = [], []
    total = len(args.policies) * len(args.budgets) * len(args.seeds)
    done = 0
    for policy in args.policies:
        for k in args.budgets:
            for seed in args.seeds:
                set_seed(seed)
                r, srows = run_sim(policy, k, X, y, t, train_end, psi_z,
                                   novelty_all, seed)
                rows.append(r)
                step_records += srows
                done += 1
                print(f"[{done:3d}/{total}] {policy:>10} k={k:3d} seed={seed} "
                      f"| caught={r['caught_post']:3d} "
                      f"| pos_labels={r['pos_labels_acquired_post']:3d} "
                      f"| PR-AUC={r['mean_pr_auc_post']:.3f}")

    # ---- append results across invocations ----
    res_csv = metrics_dir / "frontier_results.csv"
    pd.DataFrame(rows).to_csv(res_csv, mode="a",
                              header=not res_csv.exists(), index=False)
    step_csv = metrics_dir / "frontier_step_records.csv"
    pd.DataFrame(step_records).to_csv(step_csv, mode="a",
                                      header=not step_csv.exists(), index=False)

    # ---- aggregate + figures from the FULL accumulated data ----
    res_all = pd.read_csv(res_csv).drop_duplicates(
        subset=["policy", "k", "seed"], keep="last")
    steps_all = pd.read_csv(step_csv).drop_duplicates(
        subset=["policy", "k", "seed", "time_step"], keep="last")

    agg = res_all.groupby(["policy", "k"]).agg(
        caught_mean=("caught_post", "mean"),
        caught_std=("caught_post", "std"),
        pos_mean=("pos_labels_acquired_post", "mean")).round(2)
    print("\n===== FRONTIER (post-drift caught, mean over seeds, all runs) =====")
    print(agg.to_string())

    # Analytic random-investigation baseline: E[caught] = sum_s n_illicit_s * k/n_s
    post_steps = [(s, ((t == s) & (y != -1)).sum(),
                   int(((t == s) & (y == 1)).sum())) for s in range(43, 50)]
    ks_line = sorted(res_all["k"].unique())
    random_line = [sum(p * min(kk, n) / n for _, n, p in post_steps)
                   for kk in ks_line]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    piv = res_all.groupby(["policy", "k"])["caught_post"].mean().unstack()
    for policy in piv.index:
        cols = piv.loc[policy].dropna()
        axes[0].plot(cols.index, cols.values, marker="o", markersize=4,
                     label=policy)
    axes[0].plot(ks_line, random_line, "k--", linewidth=1.2,
                 label="random investigation")
    axes[0].set_xlabel("Investigation budget k (alerts/step)")
    axes[0].set_ylabel("Illicit caught, post-drift (of 169)")
    axes[0].set_title("Label-acquisition frontier")
    axes[0].legend(fontsize=7)

    sp = steps_all[steps_all["pr_auc"].notna()]
    axes[1].scatter(sp["cum_pos_before"], sp["pr_auc"],
                    s=14, alpha=0.45, c="#4878a8")
    med = sp.groupby(pd.cut(sp["cum_pos_before"],
                            bins=[-1, 0, 5, 10, 20, 40, 80, 160, 1000]),
                     observed=True)["pr_auc"].median()
    axes[1].set_xlabel("Illicit labels acquired BEFORE step")
    axes[1].set_ylabel("PR-AUC at step")
    axes[1].set_title("Mediation: model quality vs labels acquired")
    fig.tight_layout()
    out = fig_dir / "fig7_frontier.pdf"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 7 -> {out}")
    print("\nMedian PR-AUC by acquired-positives bin:")
    print(med.round(3).to_string())


if __name__ == "__main__":
    main()