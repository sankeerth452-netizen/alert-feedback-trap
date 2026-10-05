"""Shared utilities for the ICAIF'26 camera-ready experiments.
Identical model/protocol constants to experiments/run_frontier.py and
experiments/run_aegis_final.py."""
import os, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
N_FEATURES = 165
FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}
LGBM = dict(n_estimators=400, num_leaves=64, learning_rate=0.1,
            class_weight="balanced", n_jobs=1, verbose=-1)
LABEL_DELAY, ALARM_Z = 1, 2.0
MIN_POS, MIN_NEG, ADAPT_WINDOW = 5, 5, 8
COMP_TAU, COMP_EMA = 0.10, 0.6
XYT_CACHE = ROOT / "data" / "processed" / "cr_xyt.npz"


def load_xyt():
    if XYT_CACHE.exists():
        z = np.load(XYT_CACHE)
        return z["X"], z["y"], z["t"]
    raw = ROOT / "data" / "raw"
    f = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    f.columns = ["txId", "time_step"] + FEATURE_COLS
    c = pd.read_csv(raw / "elliptic_txs_classes.csv")
    c["label"] = c["class"].astype(str).map(LABEL_MAP)
    df = f.merge(c[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype(int)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    t = df["time_step"].values.astype(int)
    np.savez(XYT_CACHE, X=X, y=y, t=t)
    return X, y, t


def mean_psi(ref, cur, n_bins=10, eps=1e-4):
    d = ref.shape[1]; out = np.zeros(d)
    for j in range(d):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref[:, j], bins=edges)[0] / max(len(ref), 1)
        q = np.histogram(cur[:, j], bins=edges)[0] / max(len(cur), 1)
        p, q = np.clip(p, eps, None), np.clip(q, eps, None)
        out[j] = np.sum((p - q) * np.log(p / q))
    return float(out.mean())


def psi_z_scores(X, t, ref_end, calib_end):
    """frontier/cold-start: (34, 34); step-42 canonical: (42, 34)."""
    ref = X[t <= ref_end]
    psi = {s: mean_psi(ref, X[t == s]) for s in range(1, 50)}
    tv = np.array([psi[s] for s in range(1, calib_end + 1)])
    return {s: (psi[s] - tv.mean()) / tv.std(ddof=1) for s in psi}


def safe_ap(y_true, prob):
    y_true = np.asarray(y_true)
    if (y_true == 1).any() and (y_true == 0).any():
        return float(average_precision_score(y_true, prob))
    return np.nan


def fit_lgbm(X, y, seed, sample_weight=None):
    m = LGBMClassifier(random_state=seed, **LGBM)
    m.fit(X, y, sample_weight=sample_weight)
    return m
