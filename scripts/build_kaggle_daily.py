#!/usr/bin/env python3
"""Convert the public Kaggle Azure cost export into a per-service daily series.

Source: c.carrucciu, "Azure Subscription Costs", Kaggle (2023),
        https://www.kaggle.com/datasets/carrucciu/azure-costs  (licence: not specified)
        Real Azure Cost Analysis export, daily, one subscription, identifiers hashed.

    python scripts/build_kaggle_daily.py

Rows that look identical are KEPT: the export has only 11 of Azure's columns, so two
separate charges (e.g. two load-balancer rules on one resource) can print identically.
Dropping them would silently remove real cost.

Outputs (Dataset/clean/):
  kaggle_daily_by_service.csv  date x ServiceName (MeterCategory) cost, zero-filled
  kaggle_daily_totals.csv      daily total + calendar features
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Dataset" / "anonymized_costs.csv"
OUT = ROOT / "Dataset" / "clean"


def main() -> None:
    df = pd.read_csv(SRC)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    dups = df.duplicated()

    wide = df.pivot_table(index="date", columns="MeterCategory",
                          values="CostInBillingCurrency", aggfunc="sum", fill_value=0.0)
    wide = wide.asfreq("D", fill_value=0.0)
    long = (wide.stack().rename("cost_usd").reset_index()
            .rename(columns={"MeterCategory": "ServiceName"}))
    long.to_csv(OUT / "kaggle_daily_by_service.csv", index=False)

    totals = wide.sum(axis=1).rename("total_cost_usd").reset_index()
    totals["day_of_week"] = totals["date"].dt.dayofweek
    totals["is_weekend"] = totals["day_of_week"] >= 5
    totals.to_csv(OUT / "kaggle_daily_totals.csv", index=False)

    print(f"Rows: {len(df)}  (identical-looking rows kept: {dups.sum()}, "
          f"{df.loc[dups, 'CostInBillingCurrency'].sum():.2f} of {df['CostInBillingCurrency'].sum():.2f} cost)")
    print(f"Days: {len(wide)}  {wide.index.min().date()} -> {wide.index.max().date()}  services: {wide.shape[1]}")
    print("Services:", ", ".join(wide.sum().sort_values(ascending=False).index))


if __name__ == "__main__":
    main()
