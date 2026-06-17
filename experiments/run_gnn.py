"""Train GCN / GAT / GraphSAGE on the temporal split; save metrics + curves.

Run from anywhere:
    python experiments/run_gnn.py
    python experiments/run_gnn.py --models graphsage
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import to_undirected

from src.data.load import load_dataframe, load_edges, build_pyg_graph
from src.data.splits import temporal_masks
from src.eval.metrics import aggregate_metrics, per_timestep_f1
from src.models.gnn import GNN
from src.utils.seed import set_seed


def f1_illicit(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import f1_score
    return f1_score(y_true, (y_prob >= threshold).astype(int),
                    pos_label=1, zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/base.yaml"))
    gnn_cfg = yaml.safe_load(open("configs/gnn.yaml"))
    models_cfg, common = gnn_cfg["models"], gnn_cfg["common"]
    if args.models:
        models_cfg = {k: v for k, v in models_cfg.items() if k in args.models}
    set_seed(cfg["seed"])

    metrics_dir = Path(cfg["paths"]["results_metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    df = load_dataframe(cfg["paths"]["raw_dir"])
    edges = load_edges(cfg["paths"]["raw_dir"])
    graph = build_pyg_graph(df, edges)
    if common["undirected"]:
        graph.edge_index = to_undirected(graph.edge_index)

    train_np, val_np, test_np = temporal_masks(df, **cfg["split"])
    train_mask = torch.tensor(train_np)
    val_mask = torch.tensor(val_np)
    test_mask = torch.tensor(test_np)
    eval_mask = val_mask | test_mask

    device = torch.device("cpu")   # CPU is reliable + fast enough at this scale
    graph = graph.to(device)

    # Class weights from train distribution (handles imbalance)
    y_train = graph.y[train_mask]
    n_licit = (y_train == 0).sum().item()
    n_illicit = (y_train == 1).sum().item()
    class_weight = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float, device=device)
    print(f"Class weights (licit, illicit): {class_weight.tolist()}")

    summary_rows = []
    for name, params in models_cfg.items():
        print(f"\n--- Training {name} ---")
        set_seed(cfg["seed"])
        model = GNN(arch=name, in_dim=graph.num_node_features,
                    hidden_dim=common["hidden_dim"],
                    dropout=common["dropout"], **params).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=common["lr"],
                               weight_decay=common["weight_decay"])

        best_val_f1, best_state, patience_left = -1.0, None, common["patience"]
        for epoch in range(1, common["max_epochs"] + 1):
            model.train()
            opt.zero_grad()
            logits = model(graph.x, graph.edge_index)
            loss = F.cross_entropy(logits[train_mask], graph.y[train_mask],
                                   weight=class_weight)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                prob = F.softmax(model(graph.x, graph.edge_index), dim=1)[:, 1]
            val_f1 = f1_illicit(graph.y[val_mask].numpy(),
                                prob[val_mask].numpy())
            if val_f1 > best_val_f1:
                best_val_f1, patience_left = val_f1, common["patience"]
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_left -= 1
            if epoch % 20 == 0 or patience_left == 0:
                print(f"epoch {epoch:3d} | loss {loss.item():.4f} "
                      f"| val F1 {val_f1:.4f} | best {best_val_f1:.4f}")
            if patience_left == 0:
                print(f"Early stop at epoch {epoch}.")
                break

        # ---- Evaluate best checkpoint ----
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(graph.x, graph.edge_index), dim=1)[:, 1].numpy()

        y_np, t_np = graph.y.numpy(), graph.time_step.numpy()
        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            m = aggregate_metrics(y_np[mask.numpy()], prob[mask.numpy()])
            summary_rows.append({"model": name, "split": split_name, **m})
            print(f"{split_name}: {m}")

        em = eval_mask.numpy()
        per_timestep_f1(y_np[em], prob[em], t_np[em]).to_csv(
            metrics_dir / f"curve_{name}.csv", index=False)
        torch.save(best_state, metrics_dir / f"gnn_{name}_best.pt")

    summary = pd.DataFrame(summary_rows)
    suffix = "_".join(models_cfg.keys()) if args.models else "all"
    summary.to_csv(metrics_dir / f"gnn_summary_{suffix}.csv", index=False)
    print("\n================ SUMMARY ================")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()