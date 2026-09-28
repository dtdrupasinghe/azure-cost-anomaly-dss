"""Cost-driver attribution: which service explains an anomalous day?

Shared by the evaluation script and the Streamlit dashboard. Input is a wide
per-service matrix `W` (index = date, columns = service, values = daily cost).

Methods compared (all rank services, most likely driver first):
  largest_cost : biggest cost that day (what a plain cost report shows; common in notebooks)
  cost_delta   : biggest $ increase vs the service's trailing 7-day median
  robust_z     : biggest increase relative to the service's own normal variability
  shap_if      : SHAP values of an Isolation Forest trained on per-service deviations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

METHODS = ["largest_cost", "cost_delta", "robust_z", "shap_if"]


def service_baseline(W: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Each service's normal daily cost = median of the previous `window` days."""
    return W.shift(1).rolling(window, min_periods=3).median()


def explain_day(W: pd.DataFrame, day, top: int = 5) -> pd.DataFrame:
    """Dashboard view: per-service cost vs normal for one day, biggest increase first."""
    B = service_baseline(W)
    t = pd.DataFrame({"cost": W.loc[day], "normal": B.loc[day].fillna(W.loc[day])})
    t["increase"] = t["cost"] - t["normal"]
    rise = t["increase"].clip(lower=0)
    t["share_of_increase"] = rise / rise.sum() if rise.sum() > 0 else 0.0
    return t.sort_values("increase", ascending=False).head(top)


def shap_contributions(W: pd.DataFrame, days, seed: int = 42) -> pd.DataFrame:
    """SHAP values (rows = days, columns = services) of an Isolation Forest trained on
    per-service deviations from normal. Negative = pushes the day towards 'anomalous'."""
    D = (W - service_baseline(W)).fillna(0.0)
    model = IsolationForest(n_estimators=200, random_state=seed).fit(D.values)
    sv = shap.TreeExplainer(model).shap_values(D.loc[list(days)].values)
    return pd.DataFrame(sv, index=list(days), columns=W.columns)


def rank_drivers(W: pd.DataFrame, days, seed: int = 42) -> dict[str, dict]:
    """Return {method: {day: [services ranked, most likely driver first]}}."""
    B = service_baseline(W)
    D = (W - B).fillna(0.0)
    mad = (W.shift(1).rolling(14, min_periods=7)
           .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True))
    eps = 1e-3 * W.sum(axis=1).median()
    Z = D / (1.4826 * mad.fillna(0.0) + eps)
    S = shap_contributions(W, days, seed)

    def order(row: pd.Series, ascending: bool = False) -> list[str]:
        return list(row.sort_values(ascending=ascending).index)

    return {
        "largest_cost": {d: order(W.loc[d]) for d in days},
        "cost_delta": {d: order(D.loc[d]) for d in days},
        "robust_z": {d: order(Z.loc[d]) for d in days},
        # Negative SHAP pushes the Isolation Forest score towards "anomalous".
        "shap_if": {d: order(S.loc[d], ascending=True) for d in days},
    }
