"""Evaluation: aggregate metrics + per-time-step F1 (the drift diagnostic)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score
)


def aggregate_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1_illicit": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "precision_illicit": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_illicit": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "n": int(len(y_true)),
        "n_illicit": int((np.asarray(y_true) == 1).sum()),
    }


def per_timestep_f1(y_true, y_prob, time_steps, threshold: float = 0.5) -> pd.DataFrame:
    df = pd.DataFrame({
        "t": np.asarray(time_steps),
        "y": np.asarray(y_true),
        "pred": (np.asarray(y_prob) >= threshold).astype(int),
    })
    rows = []
    for t, g in df.groupby("t"):
        rows.append({
            "time_step": int(t),
            "f1_illicit": round(f1_score(g["y"], g["pred"], pos_label=1, zero_division=0), 4),
            "n_illicit": int((g["y"] == 1).sum()),
            "n": len(g),
        })
    return pd.DataFrame(rows)