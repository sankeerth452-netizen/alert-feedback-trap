"""Drift detection via Population Stability Index (PSI).

PSI compares the distribution of incoming data against a reference window,
feature by feature. Rule of thumb from credit-risk practice:
  PSI < 0.10 : stable | 0.10-0.25 : moderate shift | > 0.25 : major shift.
The mean PSI over features serves both as (a) AEGIS's gating signal and
(b) an online drift alarm for compliance teams.
"""
import numpy as np


def psi_per_feature(ref: np.ndarray, cur: np.ndarray,
                    n_bins: int = 10, eps: float = 1e-4) -> np.ndarray:
    """PSI for each column. ref: (n_ref, d), cur: (n_cur, d). Returns (d,)."""
    d = ref.shape[1]
    out = np.zeros(d)
    for j in range(d):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:          # (near-)constant feature: no signal
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref[:, j], bins=edges)[0] / max(len(ref), 1)
        q = np.histogram(cur[:, j], bins=edges)[0] / max(len(cur), 1)
        p, q = np.clip(p, eps, None), np.clip(q, eps, None)
        out[j] = np.sum((p - q) * np.log(p / q))
    return out


def mean_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    return float(psi_per_feature(ref, cur, n_bins).mean())