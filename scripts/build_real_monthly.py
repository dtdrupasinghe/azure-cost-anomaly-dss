#!/usr/bin/env python3
"""Consolidate the real Azure Cost Management exports into one tidy table.

Usage:
    python scripts/build_real_monthly.py

Each export (Granularity = Monthly, Group by = ResourceId) covers one month but
also spills one extra day into the next month (UsageDate = 1st of next month).
Those spill-over rows are exactly one day of cost, so they are kept separately
as real single-day observations.

Outputs (Dataset/clean/):
    real_monthly.csv      one row per month x resource x meter (full-month cost)
    real_single_days.csv  one row per day x resource x meter (real 1-day cost)
    real_summary.csv      month x service cost pivot, for EDA / thesis tables
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
OUT = DATA / "clean"

# month label -> export file. The "(1)" January copy keeps the original text dates;
# the other January copy has two cells altered to May/June dates.
EXPORTS = {
    "2026-01": "CostManagement_MCT_2026-02-20-1406 (1).xlsx",
    "2026-04": "CostManagement_MCT_Aprial.xlsx",
    "2026-05": "CostManagement_MCT_-may.xlsx",
    "2026-06-28": "CostManagement_MCT_2026-06-29-0958.xlsx",  # single-day export
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data")
    df["UsageDate"] = pd.to_datetime(df["UsageDate"].astype(str).str[:10])
    df["resource"] = df["ResourceId"].str.split("/").str[-1]
    return df


def main() -> None:
    monthly, days = [], []
    for label, name in EXPORTS.items():
        df = load(DATA / name)
        if len(label) == 10:  # single-day export: whole file is one real day
            days.append(df.assign(date=pd.Timestamp(label)))
            continue
        month_start = pd.Timestamp(label + "-01")
        monthly.append(df[df["UsageDate"] == month_start].assign(month=label))
        spill = df[df["UsageDate"] > month_start]
        days.append(spill.assign(date=spill["UsageDate"]))

    keep = ["resource", "ResourceGroupName", "ResourceType", "ServiceName",
            "ServiceTier", "Meter", "CostUSD"]
    m = pd.concat(monthly)[["month"] + keep]
    d = pd.concat(days)[["date"] + keep]

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "real_monthly.csv", index=False)
    d.to_csv(OUT / "real_single_days.csv", index=False)
    summary = m.pivot_table(index="ServiceName", columns="month", values="CostUSD",
                            aggfunc="sum", fill_value=0).round(2)
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(OUT / "real_summary.csv")

    print("Monthly cost by service (USD):\n", summary.to_string())
    per_day = d.groupby("date")["CostUSD"].sum().round(2)
    print("\nReal single days (USD):\n", per_day.to_string())
    print("\nSingle-day cost by service:\n",
          d.pivot_table(index="ServiceName", columns=d["date"].dt.date, values="CostUSD",
                        aggfunc="sum", fill_value=0).round(3).to_string())


if __name__ == "__main__":
    main()
