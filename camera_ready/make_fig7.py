"""Camera-ready Figure 7: ignition probability (fixed data vs transaction bootstrap)
and within-step mediation. Usage: python make_fig7.py <results_dir> <out_prefix>"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES, OUT = sys.argv[1], sys.argv[2]
IGN = 35
PREV = {47: 22/846, 48: 36/471, 49: 56/476}
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#444444"})
NAVY, GREEN, ORANGE = "#1f3050", "#3d7a5a", "#c0843d"

def load(exp):
    runs, steps = [], []
    for ln in open(f"{RES}/{exp}.jsonl"):
        r = json.loads(ln); kw = r["kw"]; post = [x for x in r["rows"] if x["time_step"] >= 43]
        d = dict(policy=kw.get("policy", "greedy"), k=kw.get("k", 50), seed=kw.get("seed", 42),
                 caught=sum(x["caught_inv"] for x in post))
        runs.append(d)
        for x in post:
            steps.append(dict(policy=d["policy"], **x))
    return pd.DataFrame(runs), pd.DataFrame(steps)

def wilson(m, n, z=1.96):
    p = m / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

fs, st1 = load("ignite_seeds"); fb, _ = load("ignite_boot"); _, st2 = load("mediation")
fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw=dict(width_ratios=[1.15, 1]))
for df, pol, col, ls, lab in [(fs, "greedy", NAVY, "-", "greedy"),
                              (fs, "eps10", GREEN, "-", r"$\epsilon$=0.1"),
                              (fb, "greedy", NAVY, "--", "greedy (bootstrap)"),
                              (fb, "eps10", GREEN, "--", r"$\epsilon$=0.1 (bootstrap)")]:
    g = df[df.policy == pol].groupby("k").caught
    k = np.array(sorted(g.groups)); n = g.size().values; m = g.apply(lambda v: (v >= IGN).sum()).values
    a.plot(k, m / n, ls=ls, color=col, lw=1.1, marker="o" if ls == "-" else None, ms=2.2, label=lab if ls == "-" else None)
    if (n > 1).all():
        lo, hi = wilson(m, n); a.fill_between(k, lo, hi, color=col, alpha=0.10, lw=0)
a.set_xlabel(r"Investigation budget $k$ / step"); a.set_ylabel(r"P(caught $\geq$ 35)")
a.set_ylim(-0.03, 1.03); a.set_xticks([50, 60, 70, 80, 90, 100]); a.grid(alpha=0.3, lw=0.4)
a.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
st = pd.concat([st1, st2]).dropna(subset=["pr_auc"])
for t, shade in ((47, "#e3b98a"), (48, ORANGE), (49, "#7a4e1c")):
    h = st[st.time_step == t]
    gb = h.groupby(pd.cut(h.cum_pos_before, [-1, 2, 5, 10, 20, 40]), observed=True)
    lift = (gb.pr_auc.median() / PREV[t])[gb.size() >= 20]
    x = [{2: 0, 5: 1, 10: 2, 20: 3, 40: 4}[int(iv.right)] for iv in lift.index]
    b.plot(x, lift.values, marker="o", ms=2.2, lw=1.1, color=shade, label=f"step {t}")
b.axhline(1, color="#8e9aaf", lw=0.7, ls="--")
b.set_xticks(range(5)); b.set_xticklabels(["0-2", "3-5", "6-10", "11-20", "21-40"], fontsize=5.0)
b.set_xlabel("Illicit labels acquired"); b.set_ylabel("PR-AUC / base rate")
b.grid(alpha=0.3, lw=0.4); b.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
for ax in (a, b):
    ax.tick_params(labelsize=5.8, width=0.5, length=2)
fig.tight_layout(pad=0.25, w_pad=0.6)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=400)
