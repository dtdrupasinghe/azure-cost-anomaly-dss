#!/usr/bin/env python3
"""Ingest an Azure Cost Management export and build a daily cost time series.

Usage:
    python scripts/ingest_daily.py <path-to-export.csv|.xlsx>

Validates that the file has true *daily* granularity (the research methodology
needs day-of-week features and daily spike injection), then builds and saves a
clean daily time series with engineered calendar features.

Output: writes `<input-stem>_daily_timeseries.csv` next to the input file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Azure exports vary; accept either of these as the date column.
DATE_CANDIDATES = ["UsageDate", "Date", "UsageDateTime"]
COST_CANDIDATES = ["CostUSD", "Cost", "PreTaxCost"]


def _pick(cols: list[str], candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(
        f"ERROR: no {label} column found. Looked for {candidates}; file has {list(cols)}"
    )


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        # Azure xlsx exports put the data on the 2nd sheet (1st is the scope header).
        xls = pd.ExcelFile(path)
        sheet = xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]).expanduser()
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    df = load(path)
    date_col = _pick(list(df.columns), DATE_CANDIDATES, "date")
    cost_col = _pick(list(df.columns), COST_CANDIDATES, "cost")

    # Azure sometimes stores the date as an Excel serial or YYYYMMDD int — coerce robustly.
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        dates = pd.to_datetime(df[date_col], origin="1899-12-30", unit="D", errors="coerce")
    df[date_col] = dates
    df = df.dropna(subset=[date_col])
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)

    n_days = df[date_col].dt.normalize().nunique()
    print(f"File:            {path.name}")
    print(f"Rows:            {len(df)}")
    print(f"Date column:     {date_col}")
    print(f"Cost column:     {cost_col}")
    print(f"Distinct dates:  {n_days}")
    print(f"Date range:      {df[date_col].min().date()} -> {df[date_col].max().date()}")

    if n_days < 7:
        print(
            "\n  WARNING: fewer than 7 distinct dates. This export is almost certainly "
            "monthly/aggregate, NOT daily. Re-export with Granularity = Daily "
            "(see docs/azure-daily-export-steps.md) before modeling.\n"
        )

    # Daily total cost time series + calendar features for the model.
    daily = (
        df.groupby(df[date_col].dt.normalize())[cost_col]
        .sum()
        .rename("total_cost_usd")
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek          # 0 = Monday
    daily["day_name"] = daily["date"].dt.day_name()
    daily["is_weekend"] = daily["day_of_week"] >= 5
    daily["day_of_month"] = daily["date"].dt.day

    out = path.with_name(path.stem + "_daily_timeseries.csv")
    daily.to_csv(out, index=False)
    print(f"\nDaily time series written to: {out}")
    print(daily.to_string(index=False))


if __name__ == "__main__":
    main()
