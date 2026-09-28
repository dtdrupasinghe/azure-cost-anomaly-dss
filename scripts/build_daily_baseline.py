#!/usr/bin/env python3
"""Build a clean DAILY baseline from the existing (monthly) Azure exports.

The raw Azure exports in `Dataset/` are monthly/aggregate, not daily. This script
takes the dated "Usage" export, selects the most complete calendar month, and
distributes each line-item's monthly cost evenly across the days of that month to
produce a realistic flat daily baseline (a running VM/disk costs ~the same each
day). The result is the "normal" pattern that the augmentation step later perturbs
with noise and injected spikes.

    python scripts/build_daily_baseline.py

Outputs (into Dataset/clean/):
  - daily_baseline.csv         one row per (date, resource, service, meter)
  - daily_baseline_totals.csv  daily total cost time series + calendar features

Design notes / assumptions (documented for the thesis):
  * Granularity gap: source has only monthly buckets, so daily values are an even
    split of the month total. This is an explicit, defensible modelling choice for
    the cold-start scenario; replace with a real daily export when available.
  * The truncated/partial February bucket is excluded (it is not a full month).
  * Sensitive identifiers (subscription GUID in ResourceId, SubscriptionId) are
    dropped; only the human-readable resource-group label is kept.
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
OUT = DATA / "clean"
# Dated "Usage" export (has the UsageDate column).
SOURCE = DATA / "CostManagement_MCT_2026-02-20-1406 (1).xlsx"

DATE_COL = "UsageDate"
COST_COL = "CostUSD"
KEEP = ["ResourceGroupName", "ServiceName", "ServiceTier", "Meter"]


def load_usage(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    # Coerce date (handles both YYYY-MM-DD text and Excel serials).
    d = pd.to_datetime(df[DATE_COL], errors="coerce")
    if d.isna().all():
        d = pd.to_datetime(df[DATE_COL], origin="1899-12-30", unit="D", errors="coerce")
    df[DATE_COL] = d
    df = df.dropna(subset=[DATE_COL])
    df[COST_COL] = pd.to_numeric(df[COST_COL], errors="coerce").fillna(0.0)
    return df


def pick_complete_month(df: pd.DataFrame) -> pd.Period:
    """Choose the month with the highest total cost = the most complete one."""
    months = df.groupby(df[DATE_COL].dt.to_period("M"))[COST_COL].sum()
    return months.idxmax()


def expand_to_daily(month_df: pd.DataFrame, period: pd.Period) -> pd.DataFrame:
    year, month = period.year, period.month
    n_days = calendar.monthrange(year, month)[1]
    days = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")

    rows = month_df[KEEP + [COST_COL]].copy()
    rows[KEEP] = rows[KEEP].fillna("None")
    # Even split of the monthly cost across every day of the month.
    rows["daily_cost_usd"] = rows[COST_COL] / n_days

    # Cross-join line-items x days.
    rows["_k"] = 1
    cal = pd.DataFrame({"date": days, "_k": 1})
    long = rows.merge(cal, on="_k").drop(columns=["_k", COST_COL])

    long = long.rename(columns={"ResourceGroupName": "resource"})
    long["day_of_week"] = long["date"].dt.dayofweek
    long["day_name"] = long["date"].dt.day_name()
    long["is_weekend"] = long["day_of_week"] >= 5
    long["day_of_month"] = long["date"].dt.day
    cols = [
        "date", "resource", "ServiceName", "ServiceTier", "Meter",
        "daily_cost_usd", "day_of_week", "day_name", "is_weekend", "day_of_month",
    ]
    return long[cols].sort_values(["date", "resource", "ServiceName"]).reset_index(drop=True)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"ERROR: source not found: {SOURCE}")
    OUT.mkdir(exist_ok=True)

    df = load_usage(SOURCE)
    period = pick_complete_month(df)
    month_df = df[df[DATE_COL].dt.to_period("M") == period]

    monthly_total = month_df[COST_COL].sum()
    daily = expand_to_daily(month_df, period)

    # Sanity: the expanded daily costs must re-sum to the original monthly total.
    assert np.isclose(daily["daily_cost_usd"].sum(), monthly_total), "split lost money!"

    totals = (
        daily.groupby("date")["daily_cost_usd"].sum()
        .rename("total_cost_usd").reset_index()
    )
    totals["day_of_week"] = totals["date"].dt.dayofweek
    totals["day_name"] = totals["date"].dt.day_name()
    totals["is_weekend"] = totals["day_of_week"] >= 5
    totals["day_of_month"] = totals["date"].dt.day

    daily.to_csv(OUT / "daily_baseline.csv", index=False)
    totals.to_csv(OUT / "daily_baseline_totals.csv", index=False)

    print(f"Source month chosen:  {period}  (most complete)")
    print(f"Line-items:           {len(month_df)}")
    print(f"Monthly total:        ${monthly_total:.2f}")
    print(f"Days expanded:        {totals.shape[0]}")
    print(f"Flat daily total:     ${totals['total_cost_usd'].iloc[0]:.4f}/day")
    print(f"Baseline rows:        {len(daily)}  (line-items x days)")
    print(f"\nWrote:\n  {OUT/'daily_baseline.csv'}\n  {OUT/'daily_baseline_totals.csv'}")
    print("\nDaily totals time series:")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
