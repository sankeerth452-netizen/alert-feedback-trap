"""Ignition analysis: bimodality + ignition probability from frontier results."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import yaml

IGNITED = 35   # caught_post threshold separating the two observed clusters

cfg = yaml.safe_load(open("configs/base.yaml"))
mdir = Path(cfg["paths"]["results_metrics"])
res = pd.read_csv(mdir / "frontier_results.csv").drop_duplicates(
    subset=["policy", "k", "seed"], keep="last")

sub = res[res.policy.isin(["greedy", "eps10", "eps20", "eps40"]) & (res.k <= 100)]
ign = sub.assign(ignited=sub.caught_post >= IGNITED).groupby(
    ["policy", "k"]).agg(n=("ignited", "size"),
                         p_ignite=("ignited", "mean"),
                         caught_mean=("caught_post", "mean")).round(2)
print(ign.to_string())

fig, ax = plt.subplots(figsize=(9, 4.5))
for policy, marker in [("greedy", "s"), ("eps10", "o")]:
    g = sub[sub.policy == policy]
    ax.scatter(g.k + (0 if policy == "greedy" else 1.5), g.caught_post,
               s=26, alpha=0.7, marker=marker, label=policy)
ax.axhline(IGNITED, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("Investigation budget k"); ax.set_ylabel("Illicit caught post-drift")
ax.set_title("Bimodal outcomes near the ignition region (each point = one run)")
ax.legend()
fig.tight_layout()
out = Path(cfg["paths"]["results_figures"]) / "fig8_ignition_bimodality.pdf"
fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".png"), dpi=200)
print(f"Saved Figure 8 -> {out}")