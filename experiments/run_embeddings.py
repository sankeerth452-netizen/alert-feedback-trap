"""Extract GraphSAGE penultimate embeddings for ALL nodes -> data/processed/.

These 128-dim structural embeddings become AEGIS's graph features.
Run AFTER run_gnn.py (needs gnn_graphsage_best.pt).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from torch_geometric.utils import to_undirected

from src.data.load import load_dataframe, load_edges, build_pyg_graph
from src.models.gnn import GNN
from src.utils.seed import set_seed


def main():
    cfg = yaml.safe_load(open("configs/base.yaml"))
    common = yaml.safe_load(open("configs/gnn.yaml"))["common"]
    set_seed(cfg["seed"])

    df = load_dataframe(cfg["paths"]["raw_dir"])
    graph = build_pyg_graph(df, load_edges(cfg["paths"]["raw_dir"]))
    if common["undirected"]:
        graph.edge_index = to_undirected(graph.edge_index)

    ckpt = Path(cfg["paths"]["results_metrics"]) / "gnn_graphsage_best.pt"
    model = GNN(arch="graphsage", in_dim=graph.num_node_features,
                hidden_dim=common["hidden_dim"], dropout=common["dropout"])
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    with torch.no_grad():
        _, h = model(graph.x, graph.edge_index, return_embedding=True)
    emb = h.numpy().astype(np.float32)

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sage_embeddings.npy", emb)
    print(f"Saved embeddings {emb.shape} -> {out_dir / 'sage_embeddings.npy'}")
    print(f"Row order == row order of features CSV (txId order preserved).")


if __name__ == "__main__":
    main()