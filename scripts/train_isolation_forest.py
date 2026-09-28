#!/usr/bin/env python3
"""Train + evaluate an Isolation Forest cost-anomaly detector (proposal RQ2 & §4.6).

Unsupervised model: trains on the augmented daily time series WITHOUT using the
labels, then evaluates its predictions against the ground-truth `is_anomaly`
column (confusion matrix, precision, recall, F1) — exactly the proposal's
evaluation plan.

    python scripts/train_isolation_forest.py

Input:
  Dataset/clean/augmented_daily_totals.csv

Outputs (results/):
  metrics.md            confusion matrix + precision/recall/F1 (for the thesis)
  predictions.csv       per-day score + predicted/actual labels
  anomaly_timeline.png  time series with TP / FP / FN highlighted

Features fed to the model (defensible, not label-leaking):
  total_cost_usd, deviation from trailing 7-day median, day_of_week, is_weekend.
The model never sees `is_anomaly`; that is used only for scoring afterwards.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset" / "clean" / "augmented_daily_totals.csv"
OUT = ROOT / "results"

SEED = 42
ROLL_WINDOW = 7
# Unsupervised contamination: our best prior guess at the anomaly rate. Set from
# domain knowledge, NOT from the labels (kept honest for the write-up).
CONTAMINATION = 0.09


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    roll_med = df["total_cost_usd"].rolling(ROLL_WINDOW, min_periods=1).median()
    df["dev_from_median"] = df["total_cost_usd"] - roll_med
    return df


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"ERROR: run augment_dataset.py first; missing {DATA}")
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(DATA, parse_dates=["date"])
    df = build_features(df)

    feature_cols = ["total_cost_usd", "dev_from_median", "day_of_week", "is_weekend"]
    X = df[feature_cols].astype(float).values
    y_true = df["is_anomaly"].astype(int).values  # used ONLY for evaluation

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=SEED,
    )
    model.fit(X)  # unsupervised — labels not passed

    raw = model.predict(X)               # -1 = anomaly, 1 = normal
    y_pred = (raw == -1).astype(int)
    df["anomaly_score"] = -model.score_samples(X)  # higher = more anomalous
    df["predicted"] = y_pred

    # ---- Evaluation -------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (tp + tn) / cm.sum()

    report = classification_report(
        y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0
    )

    # ---- Persist results --------------------------------------------------
    df[["date", "total_cost_usd", "anomaly_score", "predicted", "is_anomaly"]].to_csv(
        OUT / "predictions.csv", index=False
    )

    md = f"""# Isolation Forest — Evaluation Results

Model: `IsolationForest(n_estimators=200, contamination={CONTAMINATION}, random_state={SEED})`
Features: {", ".join(feature_cols)}
Days evaluated: {len(df)}  |  True anomalies: {int(y_true.sum())}

## Confusion Matrix

|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| **Actual Normal**  | TN = {tn} | FP = {fp} |
| **Actual Anomaly** | FN = {fn} | TP = {tp} |

## Metrics

| Metric | Value |
|--------|-------|
| Precision | {precision:.3f} |
| Recall    | {recall:.3f} |
| F1-score  | {f1:.3f} |
| Accuracy  | {accuracy:.3f} |

## scikit-learn classification report

```
{report}
```
"""
    (OUT / "metrics.md").write_text(md)

    # ---- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df["total_cost_usd"], color="#4C78A8", lw=1, label="Daily cost (USD)")
    tp_m = (df.predicted == 1) & (df.is_anomaly == 1)
    fp_m = (df.predicted == 1) & (df.is_anomaly == 0)
    fn_m = (df.predicted == 0) & (df.is_anomaly == 1)
    ax.scatter(df.date[tp_m], df.total_cost_usd[tp_m], color="green", s=55, label="True positive", zorder=5)
    ax.scatter(df.date[fp_m], df.total_cost_usd[fp_m], color="orange", marker="^", s=55, label="False positive", zorder=5)
    ax.scatter(df.date[fn_m], df.total_cost_usd[fn_m], color="red", marker="x", s=70, label="False negative (missed)", zorder=5)
    ax.set_title("Azure Daily Cost — Isolation Forest Anomaly Detection")
    ax.set_xlabel("Date"); ax.set_ylabel("Total cost (USD)")
    ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(OUT / "anomaly_timeline.png", dpi=130)

    # ---- Console summary --------------------------------------------------
    print(f"Days: {len(df)}  True anomalies: {int(y_true.sum())}  Flagged: {int(y_pred.sum())}")
    print(f"Confusion matrix  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Accuracy={accuracy:.3f}")
    print(f"\nWrote:\n  {OUT/'metrics.md'}\n  {OUT/'predictions.csv'}\n  {OUT/'anomaly_timeline.png'}")


if __name__ == "__main__":
    main()
