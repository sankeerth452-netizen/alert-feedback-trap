"""EDA + sanity checks. Produces Figure 1 (class distribution over time).

Run from anywhere:  python experiments/run_eda.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # all relative paths in configs now resolve from project root

import matplotlib.pyplot as plt
import yaml

from src.data.load import load_dataframe, load_edges, build_pyg_graph
from src.data.splits import temporal_masks, split_summary
from src.utils.seed import set_seed


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    set_seed(cfg["seed"])
    fig_dir = Path(cfg["paths"]["results_figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(cfg["paths"]["raw_dir"])
    edges = load_edges(cfg["paths"]["raw_dir"])
    graph = build_pyg_graph(df, edges)

    # ---- Sanity checks (expected values from Weber et al., 2019) ----
    print("=" * 60)
    print(f"Total transactions : {len(df):>9}   (expected 203,769)")
    print(f"Total edges        : {len(edges):>9}   (expected 234,355)")
    print(f"Time steps         : {df.time_step.min()}-{df.time_step.max()}   (expected 1-49)")
    print(f"Illicit (label=1)  : {(df.label == 1).sum():>9}   (expected   4,545)")
    print(f"Licit   (label=0)  : {(df.label == 0).sum():>9}   (expected  42,019)")
    print(f"Unknown (label=-1) : {(df.label == -1).sum():>9}   (expected 157,205)")
    print(f"PyG graph          : {graph}")
    print("=" * 60)

    # ---- Temporal split summary ----
    masks = temporal_masks(df, **cfg["split"])
    print("\nTemporal split:")
    print(split_summary(df, masks).to_string(index=False))

    # ---- Figure 1: labeled class counts per time step ----
    counts = (
        df[df.label != -1]
        .groupby(["time_step", "label"]).size().unstack(fill_value=0)
        .rename(columns={0: "licit", 1: "illicit"})
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts["licit"], label="Licit", color="#4878a8")
    ax.bar(counts.index, counts["illicit"], bottom=counts["licit"],
           label="Illicit", color="#c44e52")
    ax.axvline(43, color="black", linestyle="--", linewidth=1.2,
               label="Dark market shutdown (t=43)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Labeled transactions")
    ax.set_title("Elliptic: labeled class distribution over time")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "fig1_class_distribution.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"\nSaved Figure 1 -> {out}")


if __name__ == "__main__":
    main()