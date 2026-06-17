"""GNN node classifiers: GCN, GAT, GraphSAGE (2-layer, full-batch)."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNN(torch.nn.Module):
    def __init__(self, arch: str, in_dim: int, hidden_dim: int,
                 num_classes: int = 2, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        self.dropout = dropout
        if arch == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
        elif arch == "gat":
            self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
            self.conv2 = GATConv(hidden_dim, num_classes, heads=1)
        elif arch == "graphsage":
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, x, edge_index, return_embedding: bool = False):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        if return_embedding:
            return out, h          # h: penultimate embeddings (AEGIS uses these later)
        return out