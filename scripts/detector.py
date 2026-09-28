"""Shared cost-anomaly detection logic (used by the trainer and the dashboard).

Keeping this in one place means the Streamlit app and the evaluation script detect
anomalies identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SEED = 42
ROLL_WINDOW = 7
FEATURE_COLS = ["total_cost_usd", "dev_from_median", "day_of_week", "is_weekend"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the daily-totals frame has the calendar + rolling features the model needs."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    if "day_of_week" not in df:
        df["day_of_week"] = df["date"].dt.dayofweek
    if "is_weekend" not in df:
        df["is_weekend"] = df["day_of_week"] >= 5
    roll_med = df["total_cost_usd"].rolling(ROLL_WINDOW, min_periods=1).median()
    df["dev_from_median"] = df["total_cost_usd"] - roll_med
    return df


TRAIL = 14          # trailing window (days) for the statistical baselines


def _trailing(s: pd.Series, window: int = TRAIL) -> pd.core.window.Rolling:
    """Rolling window over the days BEFORE each day (no look-ahead)."""
    return s.shift(1).rolling(window, min_periods=7)


def baseline_flags(totals: pd.Series, warmup: int = TRAIL) -> pd.DataFrame:
    """Simple, explainable detectors on the daily total. Each column is a 0/1 flag.

    fixed_budget  : cost > 1.3 x average of the first `warmup` days (a static budget alert)
    moving_avg    : cost > 1.2 x trailing 7-day mean (moving-average band)
    three_sigma   : cost > trailing mean + 3 x trailing std
    median_mad    : cost > trailing median + 3.5 x 1.4826 x trailing MAD (robust z > 3.5)
    """
    t = totals.reset_index(drop=True)
    budget = 1.3 * t.iloc[:warmup].mean()
    mean, std, med = _trailing(t).mean(), _trailing(t).std(), _trailing(t).median()
    mad = _trailing(t).apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    out = pd.DataFrame({
        "fixed_budget": t > budget,
        "moving_avg": t > 1.2 * t.shift(1).rolling(7, min_periods=3).mean(),
        "three_sigma": t > mean + 3 * std,
        "median_mad": t > med + 3.5 * 1.4826 * mad.clip(lower=1e-9),
    })
    return out.fillna(False).astype(int)


def detect(df: pd.DataFrame, contamination: float = 0.09) -> pd.DataFrame:
    """Fit Isolation Forest and return df with `anomaly_score` and `predicted` columns."""
    df = add_features(df)
    X = df[FEATURE_COLS].astype(float).values
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=SEED
    )
    model.fit(X)
    df["anomaly_score"] = -model.score_samples(X)   # higher = more anomalous
    df["predicted"] = (model.predict(X) == -1).astype(int)
    return df
