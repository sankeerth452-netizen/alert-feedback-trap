"""AEGIS: drift-gated ensemble of temporal LightGBM experts over hybrid
(raw + GraphSAGE) features.

Design:
- Each expert trains on labeled nodes from its own time window.
- At inference, each incoming time step is compared (PSI, label-free) against
  every expert's training-window distribution; experts are weighted by
  softmax(-PSI / temperature). The expert whose 'world' most resembles
  today gets the most votes.
- Gating uses RAW features only (X_gate): PSI on quantile bins is
  scale-robust there, and it keeps the gate independent of the GNN.
"""
import numpy as np
from lightgbm import LGBMClassifier

from src.drift.detectors import mean_psi


class AegisEnsemble:
    def __init__(self, windows, lgbm_params, temperature=0.05,
                 n_bins=10, gating="psi", seed=42):
        self.windows = [tuple(w) for w in windows]
        self.lgbm_params = lgbm_params
        self.temperature = temperature
        self.n_bins = n_bins
        self.gating = gating          # "psi" or "uniform"
        self.seed = seed

    def fit(self, X_model, X_gate, y, t):
        self.experts_ = []
        for lo, hi in self.windows:
            in_win = (t >= lo) & (t <= hi)
            labeled = in_win & (y != -1)
            clf = LGBMClassifier(random_state=self.seed, **self.lgbm_params)
            clf.fit(X_model[labeled], y[labeled])
            self.experts_.append({
                "window": (lo, hi),
                "model": clf,
                "ref": X_gate[in_win],   # ALL nodes (label-free reference)
            })
            print(f"  expert [{lo:2d},{hi:2d}] trained on "
                  f"{labeled.sum():5d} labeled nodes "
                  f"({(y[labeled] == 1).sum()} illicit)")
        return self

    def _gate(self, X_gate_step):
        psis = np.array([mean_psi(e["ref"], X_gate_step, self.n_bins)
                         for e in self.experts_])
        if self.gating == "uniform":
            w = np.full(len(psis), 1.0 / len(psis))
        else:
            logits = -psis / self.temperature
            logits -= logits.max()               # numerical stability
            w = np.exp(logits)
            w /= w.sum()
        return w, psis

    def predict_proba(self, X_model, X_gate, t):
        prob = np.zeros(len(X_model))
        self.gate_log_ = []
        for step in np.unique(t):
            m = t == step
            w, psis = self._gate(X_gate[m])
            expert_probs = np.stack(
                [e["model"].predict_proba(X_model[m])[:, 1]
                 for e in self.experts_])
            prob[m] = w @ expert_probs
            self.gate_log_.append({
                "time_step": int(step),
                **{f"w_{i}": round(float(w[i]), 4) for i in range(len(w))},
                **{f"psi_{i}": round(float(psis[i]), 4) for i in range(len(psis))},
            })
        return prob