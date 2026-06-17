"""Audit layer: generate per-alert evidence reports (SHAP + graph + drift context).

Produces:
  - results/figures/fig6_shap_summary.(pdf|png)  : global feature importance
  - results/metrics/audit_reports.json           : machine-readable reports
  - results/metrics/audit_report_sample.md       : one rendered example (paper Fig. 7)

Run from anywhere:  python experiments/run_audit.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import shap
import yaml
from lightgbm import LGBMClassifier

from src.data.load import FEATURE_COLS, load_dataframe, load_edges
from src.drift.detectors import mean_psi
from src.utils.seed import set_seed

LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
N_REPORTS = 3        # top alerts to document
TOP_FEATS = 8        # features per report


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])
    train_end = cfg["split"]["train_end"]
    metrics_dir = Path(cfg["paths"]["results_metrics"])
    fig_dir = Path(cfg["paths"]["results_figures"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    X, y, t = df[FEATURE_COLS].values, df["label"].values, df["time_step"].values

    # ---- model (base expert) ----
    bm = (t <= train_end) & (y != -1)
    model = LGBMClassifier(random_state=cfg["seed"], **LGBM).fit(X[bm], y[bm])

    # ---- adjacency for graph context ----
    edges = load_edges(cfg["paths"]["raw_dir"])
    tx_to_idx = {tx: i for i, tx in enumerate(df["txId"].values)}
    nbrs = defaultdict(list)
    for a, b in edges.values:
        ia, ib = tx_to_idx.get(a), tx_to_idx.get(b)
        if ia is not None and ib is not None:
            nbrs[ia].append(ib); nbrs[ib].append(ia)

    # ---- reference stats for percentiles ----
    licit_ref = X[bm & (y == 0)]

    # ---- drift context (calibrated PSI) ----
    ref = X[t <= train_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, train_end + 1)])
    psi_z = {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}

    # ---- SHAP ----
    explainer = shap.TreeExplainer(model)
    val_mask = (t > train_end) & (t <= cfg["split"]["val_end"]) & (y != -1)
    val_idx = np.where(val_mask)[0]
    prob = model.predict_proba(X[val_idx])[:, 1]
    alerts = val_idx[np.argsort(prob)[::-1][:N_REPORTS]]

    sv = explainer.shap_values(X[alerts])
    sv = sv[1] if isinstance(sv, list) else sv      # class-1 attributions

    # Figure 6: global importance on a val sample
    sample = X[val_idx[np.random.default_rng(cfg["seed"]).choice(
        len(val_idx), size=min(2000, len(val_idx)), replace=False)]]
    sv_glob = explainer.shap_values(sample)
    sv_glob = sv_glob[1] if isinstance(sv_glob, list) else sv_glob
    shap.summary_plot(sv_glob, sample, feature_names=FEATURE_COLS,
                      max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_shap_summary.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig6_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- per-alert reports ----
    reports = []
    for row, node in enumerate(alerts):
        order = np.argsort(np.abs(sv[row]))[::-1][:TOP_FEATS]
        evidence = []
        for j in order:
            pct = float((licit_ref[:, j] < X[node, j]).mean() * 100)
            evidence.append({
                "feature": FEATURE_COLS[j],
                "shap": round(float(sv[row, j]), 4),
                "value_percentile_vs_licit": round(pct, 1),
                "direction": "raises suspicion" if sv[row, j] > 0 else "lowers suspicion",
            })
        nb = nbrs.get(int(node), [])
        nb_lab = y[nb] if nb else np.array([])
        step = int(t[node])
        reports.append({
            "txId": int(df["txId"].iloc[node]),
            "time_step": step,
            "alert_score": round(float(model.predict_proba(X[[node]])[0, 1]), 4),
            "drift_status": {"psi_z": round(float(psi_z[step]), 2),
                             "alarm": bool(psi_z[step] >= 2.0)},
            "graph_context": {
                "n_neighbors": len(nb),
                "n_neighbors_known_illicit": int((nb_lab == 1).sum()),
                "n_neighbors_known_licit": int((nb_lab == 0).sum()),
            },
            "evidence": evidence,
            "note": ("Feature semantics anonymized in the Elliptic benchmark; "
                     "in deployment, map evidence features to FATF red-flag "
                     "indicator categories."),
        })

    with open(metrics_dir / "audit_reports.json", "w") as f:
        json.dump(reports, f, indent=2)

    # ---- rendered sample (for the paper) ----
    r = reports[0]
    md = [f"# Alert Evidence Report — tx {r['txId']}",
          f"**Time step:** {r['time_step']}   "
          f"**Alert score:** {r['alert_score']}   "
          f"**Drift alarm:** {'ACTIVE' if r['drift_status']['alarm'] else 'inactive'} "
          f"(PSI z = {r['drift_status']['psi_z']})", "",
          f"**Graph context:** {r['graph_context']['n_neighbors']} linked transactions "
          f"({r['graph_context']['n_neighbors_known_illicit']} known illicit, "
          f"{r['graph_context']['n_neighbors_known_licit']} known licit)", "",
          "| Evidence feature | SHAP | vs licit population | Direction |",
          "|---|---|---|---|"]
    for e in r["evidence"]:
        md.append(f"| {e['feature']} | {e['shap']:+.3f} | "
                  f"{e['value_percentile_vs_licit']}th pct | {e['direction']} |")
    (metrics_dir / "audit_report_sample.md").write_text("\n".join(md))

    print(f"Wrote {len(reports)} reports -> {metrics_dir / 'audit_reports.json'}")
    print(f"Sample report -> {metrics_dir / 'audit_report_sample.md'}")
    print(f"Figure 6 -> {fig_dir / 'fig6_shap_summary.pdf'}")


if __name__ == "__main__":
    main()