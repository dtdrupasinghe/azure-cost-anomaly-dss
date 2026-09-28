#!/usr/bin/env python3
"""Build a labelled 6-month daily dataset anchored on the real Azure bills.

    python scripts/build_real_monthly.py        # once: consolidate the real exports
    python scripts/augment_dataset.py [--seed 42] [--k-min 0.3] [--k-max 2.0] [--tag NAME]

Inputs (Dataset/clean/, written by build_real_monthly.py):
  real_monthly.csv      real monthly cost per line-item for Jan, Apr, May 2026
  real_single_days.csv  real one-day cost per line-item (1 Feb, 1 May, 1 Jun, 28 Jun)

Outputs (Dataset/clean/, suffixed with _<tag> when --tag is given):
  augmented_daily.csv          per line-item per day, with is_anomaly / anomaly_type / incident_id
  augmented_daily_totals.csv   daily total series + calendar features + is_anomaly
  ground_truth_incidents.csv   one row per injected incident: dates, type, driver service(s), extra $

Everything is seeded so the dataset is reproducible for the thesis.

Model (documented for the write-up)
-----------------------------------
1. Daily level per line-item, month by month, from real evidence only:
     Jan  real January bill / 31                 (real month)
     Feb  real 1-Feb single-day cost             (real day anchor)
     Mar  mean of Feb and Apr levels             (linear interpolation)
     Apr  real April bill / 30                   (real month)
     May  real May bill / 31                     (real month)
     Jun  May level carried forward              (confirmed by the real 1-Jun day)
   Items absent in a month cost 0 that month, so real service churn is preserved
   (Front Door only in April, App Service from May, VM size change Jan -> Apr).
2. Day-to-day variation, calibrated on the real bills:
     fixed-price meters (VM hours, IPs, registry, disks) ~ N(1, FIXED_SIGMA)
       -> the real single days show these are near-constant (registry = $0.167 every day)
     usage-based meters (egress, tokens, logs, e-mail, search) ~ LogNormal(mean 1, USAGE_SIGMA),
       x WEEKEND_FACTOR on weekends
3. Incidents: on chosen days, extra cost = k x that day's normal total, added to the
   driver service(s). k ~ U(k_min, k_max); the default range spans the real 28-Jun
   spike (+71% of a normal day). If the driver service does not exist that day, a new
   line-item is created, mirroring a newly deployed resource.
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "Dataset" / "clean"

YEAR = 2026
MONTHS = range(1, 7)                       # Jan .. Jun 2026
FIXED_SIGMA = 0.02                         # fixed-price meters: ~2% daily jitter
USAGE_SIGMA = 0.35                         # usage-based meters: lognormal spread
WEEKEND_FACTOR = 0.85                      # usage-based meters ~15% lower on weekends
USAGE_SERVICES = {"Bandwidth", "Foundry Models", "Log Analytics", "Email",
                  "MS Bing Services", "Azure Front Door Service"}
USAGE_METER_WORDS = ("Operations", "Tokens", "Data", "Processed")

N_INCIDENTS = 12
MIN_GAP_DAYS = 6                           # keep incidents separate events
WARMUP_DAYS = 14                           # no incidents before the detector has history

# (anomaly_type, driver services, duration range in days)
INCIDENTS = [
    ("vm_left_running",    ["Virtual Machines"],                    (1, 3)),
    ("runaway_egress",     ["Bandwidth"],                           (1, 2)),
    ("storage_growth",     ["Storage"],                             (1, 2)),
    ("runaway_llm_tokens", ["Foundry Models"],                      (1, 1)),
    ("ai_agent_rollout",   ["Foundry Models", "MS Bing Services"],  (1, 2)),  # real 28-Jun pattern
    ("new_app_deploy",     ["Azure App Service"],                   (1, 3)),
]

KEY = ["resource", "ServiceName", "ServiceTier", "Meter"]


def monthly_levels() -> pd.DataFrame:
    """Daily cost level per line-item for each month Jan..Jun (columns 1..6)."""
    m = pd.read_csv(CLEAN / "real_monthly.csv")
    d = pd.read_csv(CLEAN / "real_single_days.csv", parse_dates=["date"])
    for df in (m, d):
        df[KEY] = df[KEY].fillna("None")

    real = m.pivot_table(index=KEY, columns="month", values="CostUSD", aggfunc="sum")
    feb1 = d[d["date"] == "2026-02-01"].groupby(KEY)["CostUSD"].sum().rename("feb1")
    lv = real.join(feb1, how="outer").fillna(0.0)

    out = pd.DataFrame(index=lv.index)
    out[1] = lv["2026-01"] / 31
    out[2] = lv["feb1"]
    out[4] = lv["2026-04"] / 30
    out[3] = (out[2] + out[4]) / 2
    out[5] = lv["2026-05"] / 31
    out[6] = out[5]
    return out[list(MONTHS)].reset_index()


LEVEL_SOURCE = {1: "real_month", 2: "real_day_anchor", 3: "interpolated",
                4: "real_month", 5: "real_month", 6: "carried_forward"}


def is_usage(df: pd.DataFrame) -> pd.Series:
    by_meter = df["Meter"].astype(str).str.contains("|".join(USAGE_METER_WORDS))
    return df["ServiceName"].isin(USAGE_SERVICES) | by_meter


def build_normal(levels: pd.DataFrame, rng) -> pd.DataFrame:
    frames = []
    for month in MONTHS:
        n = calendar.monthrange(YEAR, month)[1]
        days = pd.date_range(f"{YEAR}-{month:02d}-01", periods=n, freq="D")
        items = levels[levels[month] > 0][KEY + [month]].rename(columns={month: "base_daily"})
        frames.append(items.merge(pd.DataFrame({"date": days}), how="cross")
                      .assign(level_source=LEVEL_SOURCE[month]))
    df = pd.concat(frames, ignore_index=True)

    usage = is_usage(df).to_numpy()
    weekend = (df["date"].dt.dayofweek >= 5).to_numpy()
    factor = np.where(
        usage,
        rng.lognormal(-USAGE_SIGMA**2 / 2, USAGE_SIGMA, len(df)) * np.where(weekend, WEEKEND_FACTOR, 1.0),
        1 + rng.normal(0, FIXED_SIGMA, len(df)),
    )
    df["cost_usd"] = (df["base_daily"] * factor).clip(lower=0)
    df["is_anomaly"] = 0
    df["anomaly_type"] = "none"
    df["incident_id"] = ""
    return df


def pick_starts(dates: pd.DatetimeIndex, rng, n: int = N_INCIDENTS) -> list[pd.Timestamp]:
    candidates = list(dates[WARMUP_DAYS:-3])
    starts: list[pd.Timestamp] = []
    while len(starts) < n and candidates:
        s = candidates[rng.integers(len(candidates))]
        starts.append(s)
        candidates = [c for c in candidates if abs((c - s).days) >= MIN_GAP_DAYS]
    return sorted(starts)


def inject_incidents(df: pd.DataFrame, k_range: tuple[float, float], rng,
                     catalogue=INCIDENTS, n_incidents: int = N_INCIDENTS):
    """Add incidents to a line-item frame whose `base_daily` is each row's normal cost."""
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    normal_total = df.groupby("date")["base_daily"].sum()
    truth, new_rows = [], []

    starts = pick_starts(dates, rng, n_incidents)
    # Balanced catalogue: every incident type appears, in random order.
    order = rng.permutation(np.resize(np.arange(len(catalogue)), len(starts)))
    for n, (start, t) in enumerate(zip(starts, order), 1):
        a_type, services, (dlo, dhi) = catalogue[t]
        duration = int(rng.integers(dlo, dhi + 1))
        k = float(rng.uniform(*k_range))
        iid = f"INC-{n:02d}"
        window = [d for d in pd.date_range(start, periods=duration, freq="D") if d in normal_total.index]
        # Split the extra cost across driver services (first driver gets the larger share).
        shares = [1.0] if len(services) == 1 else [0.7, 0.3]
        extra_total = 0.0

        for day in window:
            extra_day = k * normal_total[day]
            extra_total += extra_day
            for svc, share in zip(services, shares):
                mask = (df["date"] == day) & (df["ServiceName"] == svc)
                amount = extra_day * share
                if mask.any():
                    w = df.loc[mask, "base_daily"]
                    w = w / w.sum() if w.sum() > 0 else 1 / mask.sum()
                    df.loc[mask, "cost_usd"] += amount * w
                    df.loc[mask, ["is_anomaly", "anomaly_type", "incident_id"]] = [1, a_type, iid]
                else:  # service not deployed that day -> a new resource appears
                    new_rows.append({"date": day, "resource": f"new-{svc.lower().replace(' ', '-')}",
                                     "ServiceName": svc, "ServiceTier": "None", "Meter": "incident",
                                     "base_daily": 0.0, "level_source": "incident",
                                     "cost_usd": amount, "is_anomaly": 1,
                                     "anomaly_type": a_type, "incident_id": iid})
        truth.append({"incident_id": iid, "start": window[0].date(), "end": window[-1].date(),
                      "duration_days": len(window), "anomaly_type": a_type,
                      "driver_primary": services[0],
                      "driver_secondary": services[1] if len(services) > 1 else "",
                      "k_multiplier": round(k, 3), "extra_cost_usd": round(extra_total, 4)})

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df, pd.DataFrame(truth)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"] >= 5
    df["day_of_month"] = df["date"].dt.day
    return df


def validate_against_real(totals: pd.DataFrame) -> None:
    """Where do the real single days fall in the synthetic normal range of their month?"""
    real = pd.read_csv(CLEAN / "real_single_days.csv", parse_dates=["date"])
    real = real.groupby("date")["CostUSD"].sum()
    normal = totals[totals["is_anomaly"] == 0]
    print("\nValidation: real single days vs synthetic normal days of the same month")
    for day, cost in real.items():
        same = normal[normal["date"].dt.month == day.month]["total_cost_usd"]
        if same.empty:
            continue
        pct = (same < cost).mean() * 100
        note = "  <- used as Feb anchor" if day == pd.Timestamp("2026-02-01") else ""
        print(f"  {day.date()}  real ${cost:.2f}  synthetic median ${same.median():.2f}"
              f"  range ${same.min():.2f}-${same.max():.2f}  percentile {pct:.0f}%{note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-min", type=float, default=0.3)
    ap.add_argument("--k-max", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="suffix for output files (experiment runs)")
    args = ap.parse_args()
    if not (CLEAN / "real_monthly.csv").exists():
        raise SystemExit("ERROR: run scripts/build_real_monthly.py first")

    rng = np.random.default_rng(args.seed)
    df = build_normal(monthly_levels(), rng)
    df, truth = inject_incidents(df, (args.k_min, args.k_max), rng)
    df = add_calendar(df)

    sfx = f"_{args.tag}" if args.tag else ""
    cols = ["date", "resource", "ServiceName", "ServiceTier", "Meter", "cost_usd",
            "is_anomaly", "anomaly_type", "incident_id", "level_source",
            "day_of_week", "day_name", "is_weekend", "day_of_month"]
    df = df[cols].sort_values(["date", "ServiceName", "resource"]).reset_index(drop=True)
    # Anonymise the case organisation's resource names in the shared output (mapped after
    # generation, so the random draws and results are unchanged).
    names = sorted(df["resource"].unique())
    df["resource"] = df["resource"].map({n: f"res-{i:02d}" for i, n in enumerate(names, 1)})
    df.to_csv(CLEAN / f"augmented_daily{sfx}.csv", index=False)

    totals = (df.groupby("date")
              .agg(total_cost_usd=("cost_usd", "sum"), is_anomaly=("is_anomaly", "max"),
                   incident_id=("incident_id", "max"), level_source=("level_source", "first"))
              .reset_index())
    totals = add_calendar(totals)
    totals.to_csv(CLEAN / f"augmented_daily_totals{sfx}.csv", index=False)
    truth.to_csv(CLEAN / f"ground_truth_incidents{sfx}.csv", index=False)

    normal = totals[totals["is_anomaly"] == 0]
    print(f"Seed {args.seed} | k in [{args.k_min}, {args.k_max}] | "
          f"{totals['date'].min().date()} -> {totals['date'].max().date()} ({len(totals)} days)")
    print(f"Anomaly days: {int(totals['is_anomaly'].sum())} from {len(truth)} incidents")
    print("\nNormal daily total by month (median USD):")
    print(normal.groupby(normal["date"].dt.strftime("%Y-%m"))
          .agg(median=("total_cost_usd", "median"), source=("level_source", "first"))
          .round(2).to_string())
    print("\nGround-truth incidents:")
    print(truth.to_string(index=False))
    validate_against_real(totals)


if __name__ == "__main__":
    main()
