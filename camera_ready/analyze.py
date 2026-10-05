"""Turn camera_ready/results/*.jsonl into the numbers/tables/figures for the
camera-ready. Usage:  python camera_ready/analyze.py [results_dir] [out_dir]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
OUTD = Path(sys.argv[2]) if len(sys.argv) > 2 else RES / "analysis"
OUTD.mkdir(parents=True, exist_ok=True)
IGN = 35                                  # paper's ignition threshold (caught_inv >= 35)
PREV = {43: 24/1370, 44: 24/1591, 45: 5/1221, 46: 2/712, 47: 22/846, 48: 36/471, 49: 56/476}
lines = []
def say(s=""):
    print(s); lines.append(s)


def load(exp):
    f = RES / f"{exp}.jsonl"
    if not f.exists():
        return None, None
    runs, steps = [], []
    for ln in open(f):
        r = json.loads(ln); kw = r["kw"]
        post = [x for x in r["rows"] if x["time_step"] >= 43]
        d = dict(kw); d.setdefault("policy", "greedy"); d.setdefault("retrainer", "window")
        d.setdefault("gate", "competence"); d.setdefault("feedback", "alert"); d.setdefault("k", 50)
        d.setdefault("seed", 42); d.setdefault("window", 8); d.setdefault("min_pos", 5)
        d.setdefault("tau", 0.1); d.setdefault("ema", 0.6); d.setdefault("boot_seed", -1)
        d["caught_topk"] = sum(x["caught_topk"] for x in post)
        d["caught_inv"] = sum(x["caught_inv"] for x in post)
        d["caught_explore"] = sum(x["caught_explore"] for x in post)
        d["n_illicit"] = sum(x["n_illicit"] for x in post)
        d["caught_48_49"] = sum(x["caught_inv"] for x in post if x["time_step"] >= 48)
        d["secs"] = r["secs"]
        runs.append(d)
        for x in post:
            steps.append({**{k: d[k] for k in ("train_end", "policy", "k", "seed", "boot_seed",
                                                "retrainer", "gate", "window")}, **x})
    return pd.DataFrame(runs), pd.DataFrame(steps)


def arm(r):
    if r.get("feedback") == "bulk":
        return "bulk"
    if r["gate"] == "static":
        return "frozen"
    name = {"window": "AEGIS", "window_ipw": "AEGIS+IPW", "window_ri": "AEGIS+RI",
            "cumulative": "cumulative", "cumulative_ipw": "cumulative+IPW",
            "alerts_all": "alerts-only (no history)"}[r["retrainer"]]
    if r["gate"] in ("replace", "average"):
        name = r["gate"]
    return f"{name} | {r['policy']}"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():6.1f} ± {x.std(ddof=1):4.1f}" if len(x) > 1 else f"{x.mean():6.1f}       "


# ------------------------------------------------------------------ repro
rc = RES / "repro_check.json"
if rc.exists():
    j = json.load(open(rc))
    say(f"== Reproduction check: {j['platform']} | py {j['python']} | lightgbm {j['lightgbm']} | all_match={j['all_match']}")
    for c in j["checks"]:
        say(f"   {'OK' if c['match'] else 'XX'} {c['check']:52s} exp {c['expected']:4d} got {c['got']:4d}")
    say()

# ------------------------------------------------------------------ key + cumulative + base_cmp (non-bootstrap)
frames = [load(e)[0] for e in ("key", "cumulative", "base_cmp")]
nb = pd.concat([f for f in frames if f is not None], ignore_index=True) if any(f is not None for f in frames) else None
if nb is not None:
    nb["arm"] = nb.apply(arm, axis=1)
    say("== Retrainer comparison, k=50 (non-bootstrap; mean ± sd over seeds; greedy arms are deterministic)")
    for te in (42, 34):
        g = nb[(nb.train_end == te) & (nb.k == 50)]
        say(f"-- base trained through step {te}")
        for a, h in sorted(g.groupby("arm"), key=lambda kv: -kv[1].caught_topk.mean()):
            say(f"   {a:38s} n={len(h):3d}  caught@topk {ms(h.caught_topk)}   investigated {ms(h.caught_inv)}")
    g = nb[(nb.train_end == 34) & (nb.k != 50)]
    for _, h in g.iterrows():
        say(f"   step-34 base, k={h.k}: {arm(h):30s} caught_inv={h.caught_inv}")
    say()

# ------------------------------------------------------------------ bootstrap
for exp in ("boot42", "boot34", "boot_cum", "boot_bulk"):
    b, _ = load(exp)
    if b is None:
        continue
    b["arm"] = b.apply(arm, axis=1)
    say(f"== Transaction bootstrap: {exp} (95% percentile intervals)")
    for te in sorted(b.train_end.unique(), reverse=True):
        g = b[(b.train_end == te) & (b.boot_seed >= 0)]
        fz = None
        for src in ("boot42", "boot34"):
            bb, _ = load(src)
            if bb is not None:
                f0 = bb[(bb.train_end == te) & (bb.gate == "static") & (bb.boot_seed >= 0)]
                if len(f0):
                    fz = f0.set_index("boot_seed").caught_topk
        for a, h in g.groupby("arm"):
            v = h.caught_topk.values
            s = f"   te={te} {a:34s} B={len(v):3d}  mean {v.mean():6.1f}  95% [{np.percentile(v,2.5):5.1f}, {np.percentile(v,97.5):5.1f}]"
            if fz is not None and a != "frozen":
                d = (h.set_index("boot_seed").caught_topk - fz).dropna()
                if len(d):
                    s += (f" | vs frozen: Δ {d.mean():+6.1f} [{np.percentile(d,2.5):+5.1f}, {np.percentile(d,97.5):+5.1f}]"
                          f"  P(arm<frozen)={np.mean(d < 0):.2f}")
            say(s)
    say()

# ------------------------------------------------------------------ sensitivity
s42, _ = load("sens42")
if s42 is not None:
    say("== Gate/trigger sensitivity, step-42 base, k=50 (greedy = deterministic; eps10 = mean ± sd, 10 seeds)")
    for p, default in (("min_pos", 5), ("window", 8), ("tau", 0.1), ("ema", 0.6)):
        others = {q: dv for q, dv in (("min_pos", 5), ("window", 8), ("tau", 0.1), ("ema", 0.6)) if q != p}
        g = s42.loc[np.logical_and.reduce([s42[q] == dv for q, dv in others.items()])]
        for v, h in g.groupby(p):
            gr = h[h.policy == "greedy"].caught_topk
            e = h[h.policy == "eps10"].caught_topk
            say(f"   {p:7s}={v:<5}{'*' if v == default else ' '} greedy {int(gr.iloc[0]) if len(gr) else '-':>4}   eps10 {ms(e) if len(e) else '-'}")
    say("   (* = paper default)\n")

w34, w34s = load("window34")
if w34 is not None:
    say("== Expert window length on the cold-start frontier (step-34 base): caught_inv, P(ignite)")
    for (k, w), h in w34.groupby(["k", "window"]):
        gr = h[h.policy == "greedy"].caught_inv; e = h[h.policy == "eps10"].caught_inv
        ig = (e >= IGN).sum(); lo, hi = wilson(ig, len(e))
        say(f"   k={k:3d} W={w:2d}  greedy {int(gr.iloc[0]) if len(gr) else '-':>4}  eps10 {ms(e)}  P(ign) {ig}/{len(e)} [{lo:.2f},{hi:.2f}]")
    say()

# ------------------------------------------------------------------ ignition
rows = []
for exp in ("ignite_seeds", "ignite_boot"):
    ig, igs = load(exp)
    if ig is None:
        continue
    say(f"== Ignition: {exp}  (ignited := caught_inv >= {IGN}; Wilson 95% CI)")
    for (pol, k), h in ig.groupby(["policy", "k"]):
        n, m = len(h), int((h.caught_inv >= IGN).sum()); lo, hi = wilson(m, n)
        share = (h.caught_48_49 / h.caught_inv.clip(lower=1)).median()
        rows.append(dict(exp=exp, policy=pol, k=k, n=n, ignited=m, p=m/n, lo=lo, hi=hi,
                         mean=h.caught_inv.mean(), sd=h.caught_inv.std(ddof=1) if n > 1 else 0,
                         share_48_49=share))
        say(f"   {pol:6s} k={k:3d} n={n:3d}  caught {ms(h.caught_inv)}  P(ign)={m/n:.2f} [{lo:.2f},{hi:.2f}]  median share from steps 48-49: {share:.2f}")
    if exp == "ignite_boot":
        p = ig.pivot_table(index=["boot_seed", "k"], columns="policy", values="caught_inv")
        say("   paired (same bootstrap sample) differences, mean [95% CI]:")
        for k, h in p.groupby(level="k"):
            for pol in ("eps10", "eps20"):
                if pol in h and "greedy" in h:
                    d = (h[pol] - h["greedy"]).dropna()
                    say(f"     k={k:3d} {pol}-greedy: {d.mean():+6.1f} [{np.percentile(d,2.5):+6.1f}, {np.percentile(d,97.5):+6.1f}]")
    say()
if rows:
    pd.DataFrame(rows).to_csv(OUTD / "ignition_table.csv", index=False)

# ------------------------------------------------------------------ mediation (within step)
st = [load(e)[1] for e in ("ignite_seeds", "mediation")]
st = pd.concat([s for s in st if s is not None], ignore_index=True) if any(s is not None for s in st) else None
if st is not None and len(st):
    st = st.dropna(subset=["pr_auc"]).copy()
    st["lift"] = st.pr_auc / st.prev
    st["bin"] = pd.cut(st.cum_pos_before, [-1, 0, 5, 10, 20, 40, 80, 1000])
    say("== Mediation: pooled (paper Fig. 7 right) vs within-step")
    say("   pooled median PR-AUC by acquired-positives bin, with mean step and prevalence in the bin:")
    for b, h in st.groupby("bin", observed=True):
        say(f"     {str(b):12s} n={len(h):5d} PR-AUC {h.pr_auc.median():.3f}  mean step {h.time_step.mean():.1f}  prev {h.prev.mean():.3f}  lift {h.lift.median():.2f}")
    say("   within-step Spearman(cum_pos_before, PR-AUC), by policy:")
    for t, h in st.groupby("time_step"):
        parts = []
        for pol, hh in h.groupby("policy"):
            if hh.cum_pos_before.nunique() > 2:
                parts.append(f"{pol}:{hh[['cum_pos_before','pr_auc']].corr('spearman').iloc[0,1]:+.2f}")
        say(f"     t={t} prev={PREV[t]:.3f}  " + "  ".join(parts))
    try:
        import statsmodels.formula.api as smf
        d = st[st.time_step >= 47].copy(); d["lp"] = np.log1p(d.cum_pos_before)
        m0 = smf.ols("np.log(lift) ~ lp + C(time_step) + C(policy)", d).fit(cov_type="cluster", cov_kwds={"groups": d["seed"].astype(str) + d["policy"]})
        m1 = smf.ols("np.log(lift) ~ lp * C(policy) + C(time_step)", d).fit()
        say(f"   OLS log(lift) ~ log1p(acquired) + step FE + policy FE (steps 47-49, n={len(d)}):")
        say(f"     slope on log1p(acquired) = {m0.params['lp']:.3f} (SE {m0.bse['lp']:.3f}, p={m0.pvalues['lp']:.1e})")
        pol_terms = [c for c in m0.params.index if c.startswith("C(policy)")]
        for c in pol_terms:
            say(f"     {c:32s} {m0.params[c]:+.3f} (p={m0.pvalues[c]:.2f})")
        from statsmodels.stats.anova import anova_lm
        m0b = smf.ols("np.log(lift) ~ lp + C(time_step) + C(policy)", d).fit()
        a = anova_lm(m0b, m1)
        say(f"     policy x acquired interaction: F={a.F.iloc[1]:.2f}, p={a['Pr(>F)'].iloc[1]:.2f}")
    except Exception as e:
        say(f"   (regression skipped: {e})")
    say()

open(OUTD / "summary.txt", "w").write("\n".join(lines))

# ------------------------------------------------------------------ figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if rows:
        T = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
        for ax, exp, title in zip(axes, ("ignite_seeds", "ignite_boot"),
                                  ("Paper protocol, 50 seeds", "Transaction bootstrap, 40 reps")):
            for pol, c, mk in (("greedy", "#444444", "s"), ("eps10", "#2a6fb0", "o"), ("eps20", "#c0632b", "^")):
                h = T[(T.exp == exp) & (T.policy == pol)].sort_values("k")
                if not len(h):
                    continue
                ax.plot(h.k, h.p, marker=mk, ms=3.5, lw=1.3, color=c, label=pol.replace("eps", "ε=0.").replace("ε=0.10", "ε=0.1").replace("ε=0.20", "ε=0.2"))
                if (h.n > 1).all():
                    ax.fill_between(h.k, h.lo, h.hi, color=c, alpha=0.12, lw=0)
            ax.set_title(title, fontsize=8.5); ax.set_xlabel("Investigation budget k", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        axes[0].set_ylabel(f"P(caught ≥ {IGN})", fontsize=8); axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(OUTD / "fig_ignition_ci.pdf"); fig.savefig(OUTD / "fig_ignition_ci.png", dpi=200)
    if st is not None and len(st):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for t, c in ((47, "#7a9cc6"), (48, "#2a6fb0"), (49, "#0b3c73")):
            h = st[st.time_step == t]
            gb = h.groupby(pd.cut(h.cum_pos_before, [-1, 2, 5, 10, 20, 40]), observed=True).lift
            g = gb.median()[gb.size() >= 20]
            x = [{2: 1, 5: 2, 10: 3, 20: 4, 40: 5}[int(iv.right)] for iv in g.index]
            ax.plot(x, g.values, marker="o", ms=3, color=c, label=f"step {t} (base rate {PREV[t]:.3f})")
        ax.axhline(1, color="grey", lw=0.8, ls=":")
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(["0-2", "3-5", "6-10", "11-20", "21-40"])
        ax.set_xlabel("Confirmed-illicit labels acquired before step", fontsize=8)
        ax.set_ylabel("PR-AUC / base rate", fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUTD / "fig_mediation_within_step.pdf"); fig.savefig(OUTD / "fig_mediation_within_step.png", dpi=200)
except Exception as e:
    print("figures skipped:", e)
print(f"\nwritten: {OUTD}")
