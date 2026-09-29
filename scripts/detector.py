"""Shared cost-anomaly detection logic (used by the trainer and the dashboard).

Keeping this in one place means the Streamlit app and the evaluation script detect
anomalies identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
    ew = t.shift(1).ewm(span=7, min_periods=7)
    out = pd.DataFrame({
        "fixed_budget": t > budget,
        "moving_avg": t > 1.2 * t.shift(1).rolling(7, min_periods=3).mean(),
        "three_sigma": t > mean + 3 * std,
        "median_mad": t > med + 3.5 * 1.4826 * mad.clip(lower=1e-9),
        "ewma": t > ew.mean() + 3 * ew.std(),
    })
    return out.fillna(False).astype(int)


def baseline_scores(totals: pd.Series, warmup: int = TRAIL) -> pd.DataFrame:
    """Continuous version of each statistical rule (higher = more anomalous), for ROC/PR-AUC."""
    t = totals.reset_index(drop=True)
    mean, std, med = _trailing(t).mean(), _trailing(t).std(), _trailing(t).median()
    mad = _trailing(t).apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    ew = t.shift(1).ewm(span=7, min_periods=7)
    out = pd.DataFrame({
        "fixed_budget": t / t.iloc[:warmup].mean(),
        "moving_avg": t / t.shift(1).rolling(7, min_periods=3).mean(),
        "three_sigma": (t - mean) / std.clip(lower=1e-9),
        "median_mad": (t - med) / (1.4826 * mad.clip(lower=1e-9)),
        "ewma": (t - ew.mean()) / ew.std().clip(lower=1e-9),
    })
    return out.fillna(0.0)


# --------------------------------------------------------------------------- #
# Machine-learning detectors (unsupervised, same 4 features, standardised)
# --------------------------------------------------------------------------- #
ML_MODELS = ["isolation_forest", "one_class_svm", "lof", "kmeans"]
CONTAMINATION = 0.09        # prior share of anomalous days, NOT tuned on labels


def ml_features(totals: pd.Series) -> np.ndarray:
    df = add_features(pd.DataFrame({"date": totals.index, "total_cost_usd": totals.values}))
    return df[FEATURE_COLS].astype(float).values


def ml_scores(X: np.ndarray, fit_rows: np.ndarray, score_rows: np.ndarray,
              contamination: float = CONTAMINATION, seed: int = SEED) -> tuple[dict, dict]:
    """Fit each ML model on X[fit_rows]; return ({model: scores}, {model: 0/1 flags}) for X[score_rows].

    Scores: higher = more anomalous. Flags: score above the (1 - contamination) quantile of the
    model's scores on its own training rows (how sklearn sets its decision threshold)."""
    sc = StandardScaler().fit(X[fit_rows])
    Xf, Xs = sc.transform(X[fit_rows]), sc.transform(X[score_rows])
    same = np.array_equal(fit_rows, score_rows)
    scores, train = {}, {}

    m = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(Xf)
    train["isolation_forest"], scores["isolation_forest"] = -m.score_samples(Xf), -m.score_samples(Xs)

    m = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination).fit(Xf)
    train["one_class_svm"], scores["one_class_svm"] = -m.decision_function(Xf), -m.decision_function(Xs)

    k = min(10, len(Xf) - 1)
    if same:   # scoring the training days themselves: classic LOF
        lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination).fit(Xf)
        train["lof"] = scores["lof"] = -lof.negative_outlier_factor_
    else:      # scoring new days: novelty mode
        lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination, novelty=True).fit(Xf)
        train["lof"], scores["lof"] = -lof.negative_outlier_factor_, -lof.score_samples(Xs)

    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(Xf)
    train["kmeans"] = km.transform(Xf).min(axis=1)
    scores["kmeans"] = km.transform(Xs).min(axis=1)

    flags = {n: (scores[n] > np.quantile(train[n], 1 - contamination)).astype(int) for n in scores}
    return scores, flags


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


# Final detector used by the DSS, chosen by the evaluation (best F1 averaged over both
# datasets): a day is anomalous when at least 2 of these 4 detectors agree. Fixed budget
# is excluded (it floods alerts when spend grows).
CONSENSUS_MEMBERS = ["median_mad", "moving_avg", "three_sigma", "isolation_forest"]
CONSENSUS_MIN_VOTES = 2


def all_flags(totals: pd.Series, with_scores: bool = False):
    """Every detector's 0/1 flag for a daily-total series, plus the `consensus` detector.
    With `with_scores`, also return a matching frame of continuous anomaly scores."""
    flags = baseline_flags(totals)
    iso = detect(pd.DataFrame({"date": totals.index, "total_cost_usd": totals.values}))
    flags.insert(0, "isolation_forest", iso["predicted"].values)
    rows = np.arange(len(totals))
    ml_s, ml_f = ml_scores(ml_features(totals), rows, rows)
    for n in ("kmeans", "lof", "one_class_svm"):
        flags.insert(1, n, ml_f[n])
    flags.index = totals.index
    flags["votes"] = flags[CONSENSUS_MEMBERS].sum(axis=1)
    flags.insert(0, "consensus", (flags["votes"] >= CONSENSUS_MIN_VOTES).astype(int))
    if not with_scores:
        return flags
    scores = baseline_scores(totals)
    scores.insert(0, "isolation_forest", iso["anomaly_score"].values)
    for n in ("kmeans", "lof", "one_class_svm"):
        scores.insert(1, n, ml_s[n])
    scores.index = totals.index
    scores.insert(0, "consensus", flags["votes"] + scores[CONSENSUS_MEMBERS].rank(pct=True).mean(axis=1) / 10)
    return flags, scores
