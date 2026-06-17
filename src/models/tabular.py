"""Tabular baselines behind one unified interface: fit(X, y) / predict_proba(X)."""
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        # LR needs scaled features; trees don't.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=seed, **params),
        )
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "xgboost":
        return xgb.XGBClassifier(random_state=seed, **params)
    if name == "lightgbm":
        return lgb.LGBMClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown model: {name}")