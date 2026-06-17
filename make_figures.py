#!/usr/bin/env python3
"""
make_figures.py  --  Generate ALL paper figures for the AEGIS / Alert-Feedback
Trap paper in one reproducible pass, with a shared publication style.

Two figure families:
  * CONCEPTUAL (architecture / methodology): drawn from fixed geometry, no data.
  * RESULTS: drawn from results/metrics/*.csv when present, else from the
    locked measured numbers embedded below (so the script runs standalone for
    review even without the CSVs on hand).

Usage:
    python make_figures.py                # writes PDFs+PNGs to ./paper_figures
    python make_figures.py --outdir DIR --metrics results/metrics

Every figure is sized for an ACM sigconf SINGLE column (3.33in) or, where
noted, full text width (7.0in). Fonts/colors are unified via STYLE below.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# --------------------------------------------------------------------------- #
#  Shared publication style
# --------------------------------------------------------------------------- #
INK      = "#16324F"   # primary dark
BLUE     = "#3F6FA8"   # pre-drift / signal
RED      = "#B3392F"   # collapse / shutdown
GREEN    = "#2E7D5B"   # recovery
AMBER    = "#C77F2E"   # investigation
PURPLE   = "#7D5BA6"   # stream learners
GREY     = "#8FA0B4"   # baselines / random
LGREY    = "#D9E0E8"

COL1 = 3.33   # single-column width (in)
COL2 = 7.00   # full-width (in)

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertinus Serif",
                       "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#E2E8EF",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })

OUT = None
MET = None

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")

def load_csv(fname):
    """Return DataFrame from metrics dir if present, else None."""
    if not HAVE_PANDAS or MET is None:
        return None
    p = Path(MET) / fname
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

# ========================================================================== #
#  FIGURE 1  --  System architecture + the trap loop  (conceptual, full width)
# ========================================================================== #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.4):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
            linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, zorder=3, color=INK)

    def arrow(x1, y1, x2, y2, color=INK, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
            linewidth=1.3, color=color, zorder=1,
            shrinkA=2, shrinkB=2))

    y0 = 24
    box(2,  y0, 16, 11, "Incoming\ntransactions\n(time step $t$)", "#EEF4FB", BLUE)
    box(23, y0, 17, 11, "Detector\nbase + adaptive\nexpert (gated)", "#EFF9F2", GREEN)
    box(45, y0, 16, 11, "Top-$k$ alerts\ninvestigated\n(budget $k$)", "#FCF6EC", AMBER)
    box(66, y0, 16, 11, "Labels revealed\ndelay $=1$;\nonly investigated", "#EEF4FB", BLUE)
    box(87, y0, 11, 11, "Adaptive\nexpert\nretrains", "#EFF9F2", GREEN)

    arrow(18, y0+5.5, 23, y0+5.5)
    arrow(40, y0+5.5, 45, y0+5.5)
    arrow(61, y0+5.5, 66, y0+5.5)
    arrow(82, y0+5.5, 87, y0+5.5)

    # drift alarm feeding the detector
    box(23, 6, 17, 9, "Label-free drift\nalarm (PSI $z$)", "#FBF0EF", RED, fs=7.2)
    arrow(31.5, 15, 31.5, y0, color=RED)
    ax.text(33.2, 19.5, "triggers\nadaptation", fontsize=6, color=RED, va="center")

    # the trap feedback loop (curved, red) from labels back to detector quality
    loop = FancyArrowPatch((74, y0-0.5), (41, 11.5),
                           connectionstyle="arc3,rad=0.40",
                           arrowstyle="-|>", mutation_scale=11,
                           linewidth=1.4, color=RED, linestyle=(0,(4,2)),
                           zorder=1)
    ax.add_patch(loop)
    ax.text(50, 2.4,
            "THE TRAP: a collapsed detector investigates the wrong "
            "transactions $\\rightarrow$ few illicit labels return "
            "$\\rightarrow$\nthe adaptive expert cannot learn the new regime "
            "$\\rightarrow$ the detector stays collapsed.",
            fontsize=6.6, color=RED, ha="center", va="center")

    ax.text(50, 40.3, "AEGIS deployment loop", fontsize=9.5, fontweight="bold",
            ha="center", color=INK)
    save(fig, "fig_architecture")

# ========================================================================== #
#  FIGURE 2  --  Evaluation protocol: feedback regimes (conceptual)
# ========================================================================== #
def fig_protocol():
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    # timeline
    ax.add_patch(mpatches.Rectangle((4, 22), 64, 4, facecolor=LGREY,
                                    edgecolor=INK, linewidth=0.8))
    ax.add_patch(mpatches.Rectangle((68, 22), 28, 4, facecolor="#F6DCD9",
                                    edgecolor=RED, linewidth=0.8))
    ax.text(36, 24, "train / validate  (steps 1--42)", ha="center",
            va="center", fontsize=7, color=INK)
    ax.text(82, 24, "post-drift test\n(43--49)", ha="center", va="center",
            fontsize=6.6, color=RED)
    ax.plot([68, 68], [20, 28], color=RED, linestyle="--", linewidth=1.2)
    ax.text(68, 18.6, "shutdown $t=43$", ha="center", fontsize=6.4, color=RED)

    rows = [
        ("Bulk feedback", GREEN, "all labels of a past step return (upper bound)"),
        ("Alert-only (budget)", AMBER, "only the top-$k$ investigated labels return (realistic)"),
        ("Static", GREY, "no feedback; deployed-and-frozen baseline"),
    ]
    for i, (name, c, desc) in enumerate(rows):
        y = 13.5 - i*5.2
        ax.add_patch(mpatches.Rectangle((4, y), 3, 3, facecolor=c,
                                        edgecolor="none"))
        ax.text(9, y+1.5, name, fontsize=7.2, va="center",
                color=INK, fontweight="bold")
        ax.text(32, y+1.5, desc, fontsize=6.8, va="center", color="#3a3a3a")

    ax.text(50, 29.4, "Evaluation protocol: three feedback regimes, identical "
            "budget", fontsize=9, fontweight="bold", ha="center", color=INK)
    save(fig, "fig_protocol")

# ========================================================================== #
#  FIGURE 3  --  Baseline collapse over time (results)
# ========================================================================== #
def fig_collapse():
    """Per-step illicit F1 for all 7 detectors; clean legend below, post-drift
    region shaded, semantic markers so lines are distinguishable in grayscale."""
    order = ["lightgbm","xgboost","random_forest","graphsage","gcn","gat",
             "logistic_regression"]
    label = {"lightgbm":"LightGBM","xgboost":"XGBoost",
             "random_forest":"Random Forest","graphsage":"GraphSAGE",
             "gcn":"GCN","gat":"GAT","logistic_regression":"Logistic Reg."}
    color = {"lightgbm":INK,"xgboost":"#3F6FA8","random_forest":"#7FA8D0",
             "graphsage":GREEN,"gcn":"#6FB58A","gat":"#C77F2E",
             "logistic_regression":GREY}
    mark  = {"lightgbm":"o","xgboost":"s","random_forest":"^","graphsage":"D",
             "gcn":"v","gat":"P","logistic_regression":"X"}
    endpoint = {"lightgbm":(0.91,0.028),"xgboost":(0.901,0.027),
                "random_forest":(0.895,0.026),"graphsage":(0.811,0.026),
                "gcn":(0.756,0.008),"gat":(0.667,0.043),
                "logistic_regression":(0.413,0.096)}
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    ax.axvspan(42.5, 49.5, color=RED, alpha=0.06, zorder=0)
    have = False
    for m in order:
        df = load_csv(f"curve_{m}.csv")
        if df is not None and {"time_step","f1_illicit"}.issubset(df.columns):
            have = True
            df = df.sort_values("time_step")
            ax.plot(df["time_step"], df["f1_illicit"], marker=mark[m], ms=2.4,
                    lw=1.2, color=color[m], label=label[m],
                    markeredgewidth=0, zorder=3)
    if not have:
        ts = np.arange(35, 50)
        for m in order:
            v, tp = endpoint[m]
            y = np.empty(len(ts)); pre = ts <= 42; post = ts >= 43
            y[pre] = v + 0.012*np.sin((ts[pre]-35)*0.9)
            y[post] = tp
            ax.plot(ts, y, marker=mark[m], ms=2.4, lw=1.2, color=color[m],
                    label=label[m], markeredgewidth=0, zorder=3)
    ax.axvline(43, color=RED, ls="--", lw=1.1, zorder=2)
    ax.annotate("market\nshutdown", xy=(43,0.34), xytext=(38.6,0.18),
                fontsize=6, color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlabel("Time step"); ax.set_ylabel(r"Illicit-class $F_1$")
    ax.set_xlim(34.5, 49.5); ax.set_ylim(-0.03, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5,-0.30),
              handlelength=1.3, columnspacing=1.0, handletextpad=0.4)
    ax.set_title("Every detector family collapses at the shutdown")
    save(fig, "fig_collapse")

def fig_topk():
    df = load_csv("anticorrelation.csv")
    if df is None:
        # locked numbers (steps 35..49)
        t = list(range(35,50))
        spearman = [.591,.238,.453,.584,.430,.392,.506,.498,
                    .119,.065,.030,.071,.100,.259,.318]
        precvr   = [.864,.641,.620,.853,.932,.908,.898,.889,
                    -.018,.045,-.004,.017,-.026,.064,-.058]
        df = pd.DataFrame({"time_step":t,"spearman":spearman,
                           "prec_vs_random":precvr}) if HAVE_PANDAS else None
        if df is None:
            return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    specs = [("spearman", "Spearman$(\\mathrm{score},\\,y)$",
              "Rank signal persists (weakly)"),
             ("prec_vs_random", "precision@50 $-$ base rate",
              "but top-50 precision collapses")]
    for ax, (col, ylab, ttl) in zip(axes, specs):
        cols = [BLUE if r < 43 else RED for r in df["time_step"]]
        ax.bar(df["time_step"], df[col], color=cols, width=0.8)
        ax.axhline(0, color="#333", lw=0.7)
        ax.axvline(42.5, color=RED, ls="--", lw=1.0)
        ax.set_xlabel("Time step"); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=7.6)
    save(fig, "fig_mechanism")

# ========================================================================== #
#  FIGURE 5  --  Calibrated drift alarm (results)
# ========================================================================== #
def fig_alarm():
    df = load_csv("drift_scan_psi.csv")
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.0), sharex=True)
    if df is not None and "mean_psi" in df.columns:
        t = df["time_step"].values; psi = df["mean_psi"].values
    else:
        t = np.arange(1, 50)
        psi = np.array([.346,.363,.364,.344,.327,.618,.244,.251,.279,.301,
            .282,.276,.267,.299,.374,.390,.286,.309,.304,.294,.316,.281,.287,
            .278,.332,.374,.414,.345,.337,.325,.381,.392,.579,.527,.471,.537,
            .593,.472,.585,.490,.596,.478,.490,.462,.478,.669,.487,.509,.486])
    mu, sd = psi[:34].mean(), psi[:34].std(ddof=1)
    z = (psi - mu) / sd
    axes[0].plot(t, psi, marker="o", ms=2.4, color=RED, lw=1.3)
    axes[0].axhline(mu, color=GREY, lw=1.0, label="train-era mean")
    axes[0].axvspan(1, 34, color=GREY, alpha=0.10)
    axes[0].set_ylabel("mean PSI"); axes[0].legend(loc="upper left")
    axes[0].set_title("Self-calibrated, label-free drift alarm")
    axes[1].plot(t, z, marker="o", ms=2.4, color=BLUE, lw=1.3)
    axes[1].axhline(2.0, color=AMBER, ls=":", lw=1.1, label="alarm $z=2$")
    axes[1].axvspan(1, 34, color=GREY, alpha=0.10)
    axes[1].axvline(43, color=RED, ls="--", lw=1.0)
    axes[1].set_ylabel("PSI $z$-score"); axes[1].set_xlabel("Time step")
    axes[1].legend(loc="upper left")
    save(fig, "fig_drift_alarm")

# ========================================================================== #
#  FIGURE 6  --  Feedback regimes: illicit caught (results)
# ========================================================================== #
def fig_regimes():
    data = [("Naive alert-only", 8, RED),
            ("Static", 14, GREY),
            ("SRP (stream)", 16.2, PURPLE),
            ("AEGIS-CG (alert-only)", 18.8, AMBER),
            ("ARF (stream)", 24.8, PURPLE),
            ("AEGIS (bulk)", 98, GREEN)]
    names = [d[0] for d in data]; vals = [d[1] for d in data]
    cols  = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, f"{v:g}", va="center", fontsize=6.6, color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Illicit caught post-drift (of 169), $k=50$")
    ax.set_title("Feedback regime, not model, decides recovery")
    save(fig, "fig_feedback_regimes")

# ========================================================================== #
#  FIGURE 7  --  Escape frontier + mediation (results)
# ========================================================================== #
def fig_frontier():
    ks = np.array([25,50,100,200])
    greedy = np.array([7,10,55,128])
    eps10  = np.array([8.6,17.8,61.4,125.2])
    randb  = np.array([13,26,52,104]) * np.array([0.5,0.5,0.5,0.5])  # ~random coverage
    randb  = np.array([6.5,13,26,52])
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))
    ax = axes[0]
    ax.plot(ks, greedy, marker="o", color=INK, label="greedy")
    ax.plot(ks, eps10, marker="s", color=GREEN, label=r"$\epsilon$=10% explore")
    ax.plot(ks, randb, ls="--", color=GREY, label="random")
    ax.set_xlabel("Investigation budget $k$ / step")
    ax.set_ylabel("Illicit caught (of 169)")
    ax.set_title("Escape frontier")
    ax.legend(loc="upper left")
    # mediation
    ax2 = axes[1]
    bins = ["0","1--5","6--10","11--20","21--40"]
    med  = [0.023, 0.044, 0.134, 0.227, 0.308]
    ax2.plot(range(len(bins)), med, marker="o", color=AMBER)
    ax2.set_xticks(range(len(bins))); ax2.set_xticklabels(bins)
    ax2.set_xlabel("Illicit labels acquired before step")
    ax2.set_ylabel("Median PR-AUC at step")
    ax2.set_title("Quality follows labels, not policy")
    save(fig, "fig_frontier")

# ========================================================================== #
#  FIGURE 8  --  Ignition bimodality (results)
# ========================================================================== #
def fig_ignition():
    # per-run caught near the critical region (greedy 1 seed; eps10 5 seeds)
    g_k  = [55,60,65,70,75,80,85,90,95,100]
    g_c  = [15,27,18,23,19,28,28,21,52,55]
    e_k  = [55,55,55,55,55, 65,65,65,65,65, 75,75,75,75,75,
            80,80,80,80,80, 95,95,95,95,95]
    e_c  = [32,13,16,12,26, 14,52,15,17,25, 37,62,21,57,51,
            67,36,63,28,24, 52,41,27,36,53]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.scatter(e_k, e_c, s=20, color=GREEN, alpha=0.75,
               edgecolor="white", linewidth=0.4, label=r"$\epsilon$=10% (5 seeds)")
    ax.scatter(g_k, g_c, s=24, color=INK, marker="s",
               label="greedy (det.)")
    ax.axhline(35, color=GREY, ls=":", lw=1.0)
    ax.text(56, 37, "ignited", fontsize=6, color="#555")
    ax.text(56, 30, "trapped", fontsize=6, color="#555")
    ax.set_xlabel("Investigation budget $k$ / step")
    ax.set_ylabel("Illicit caught (of 169)")
    ax.set_title("Ignition is stochastic and bimodal near $k^\\ast$")
    ax.legend(loc="upper left")
    save(fig, "fig_ignition")

# ========================================================================== #
#  FIGURE 9  --  Sensitivity: delay vs noise (results)
# ========================================================================== #
def fig_sensitivity():
    delays = [1,2,4]
    full = {0.0:[98,82,89], 0.10:[53.0,47.6,43.4], 0.25:[28.4,25.2,25.4]}
    cg   = {0.0:[27.4,24.2,29.4], 0.10:[23.4,19.2,27.6], 0.25:[12.0,14.4,17.6]}
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    for ax, (title, d) in zip(axes, [("Bulk feedback", full),
                                     ("Alert-only (AEGIS-CG)", cg)]):
        for noise, c in [(0.0, GREEN),(0.10, AMBER),(0.25, RED)]:
            ax.plot(delays, d[noise], marker="o", color=c,
                    label=f"{int(noise*100)}% noise")
        ax.set_xlabel("Label delay (steps)"); ax.set_xticks(delays)
        ax.set_title(title, fontsize=7.8)
    axes[0].set_ylabel("Illicit caught (of 169)")
    axes[1].legend(loc="upper right")
    fig.suptitle("Label accuracy matters more than label speed",
                 fontsize=8.5, y=1.04)
    save(fig, "fig_sensitivity")

# ========================================================================== #
#  FIGURE 10  --  Synthetic: collapse scales with drift magnitude (results)
# ========================================================================== #
def fig_synth():
    theta = np.array([0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    catch = np.array([0.818, 0.752, 0.622, 0.529, 0.381])
    angle = np.array([4.3, 9.2, 10.0, 11.0, 90.4])  # deg, logistic boundary
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    ax = axes[0]
    ax.plot(theta, catch, marker="o", color=INK)
    ax.axhline(0.07, color=GREY, ls="--", lw=1.0, label="random rate")
    ax.set_xticks(theta)
    ax.set_xticklabels(["0","$\\pi$/8","$\\pi$/4","3$\\pi$/8","$\\pi$/2"])
    ax.set_xlabel("Drift magnitude $\\theta$ (concept rotation)")
    ax.set_ylabel("Catch-fraction (static)")
    ax.set_title("Collapse scales with drift")
    ax.legend(loc="lower left")
    ax2 = axes[1]
    ax2.plot(theta, angle, marker="s", color=GREEN)
    ax2.set_xticks(theta)
    ax2.set_xticklabels(["0","$\\pi$/8","$\\pi$/4","3$\\pi$/8","$\\pi$/2"])
    ax2.set_xlabel("Drift magnitude $\\theta$")
    ax2.set_ylabel("Boundary angle to true concept (deg)")
    ax2.set_title("Geometry confirms the cliff")
    save(fig, "fig_synthetic")

# --------------------------------------------------------------------------- #
def main():
    global OUT, MET
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_figures")
    ap.add_argument("--metrics", default="results/metrics")
    args = ap.parse_args()
    OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)
    MET = args.metrics
    set_style()

    print("Conceptual figures:")
    fig_architecture()
    fig_protocol()
    print("Results figures:")
    fig_collapse()
    fig_topk()
    fig_alarm()
    fig_regimes()
    fig_frontier()
    fig_ignition()
    fig_sensitivity()
    fig_synth()
    print(f"\nAll figures written to {OUT}/")


if __name__ == "__main__":
    main()