#!/usr/bin/env python3
"""Full evaluation: detection vs baselines, and cost-driver attribution accuracy.

    python scripts/build_real_monthly.py && python scripts/build_kaggle_daily.py
    python scripts/evaluate.py

Two datasets, each with incidents of KNOWN cause injected at controlled sizes:
  startup : 181-day series anchored on the case organisation's real Azure bills
            (scripts/augment_dataset.py), 12 incidents per run
  kaggle  : REAL 89-day Azure daily billing (public Kaggle export) with 6 incidents
            injected on top of the real day-to-day behaviour

Incident size k = extra cost as a fraction of a normal day's total
(k = 0.3 -> the day costs ~30% more). Swept over K_LEVELS x SEEDS.

Outputs (results/evaluation/):
  detection_runs.csv, detection_summary.csv      per detector, dataset, k
  attribution_runs.csv, attribution_summary.csv  per method, dataset, k
  fig_detection_vs_size.png, fig_attribution.png
  real_cases.md                                  unlabelled REAL spikes explained
  REPORT.md                                      thesis-ready tables
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
import augment_dataset as aug  # noqa: E402
from attribution import METHODS, explain_day, rank_drivers  # noqa: E402
from detector import baseline_flags, detect  # noqa: E402

CLEAN = ROOT / "Dataset" / "clean"
OUT = ROOT / "results" / "evaluation"
K_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
SEEDS = range(10)
WARMUP = 14
DETECTORS = ["isolation_forest", "median_mad", "three_sigma", "moving_avg", "fixed_budget"]

KAGGLE_INCIDENTS = [
    ("vm_left_running",     ["Virtual Machines"],                          (1, 3)),
    ("storage_growth",      ["Storage"],                                   (1, 2)),
    ("sql_scale_up",        ["SQL Database"],                              (1, 2)),
    ("log_ingestion_burst", ["Log Analytics"],                             (1, 1)),
    ("runaway_egress",      ["Bandwidth"],                                 (1, 2)),
    ("analytics_rollout",   ["Azure Synapse Analytics", "Log Analytics"],  (1, 2)),
]

# Reference palette, categorical slots 1-5 in fixed order (dataviz skill).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def kaggle_frame() -> pd.DataFrame:
    k = pd.read_csv(CLEAN / "kaggle_daily_by_service.csv", parse_dates=["date"])
    k = k.sort_values(["ServiceName", "date"])
    trailing = (k.groupby("ServiceName")["cost_usd"]
                .transform(lambda s: s.shift(1).rolling(7, min_periods=1).median()))
    return k.assign(resource=k["ServiceName"], ServiceTier="None", Meter="all",
                    base_daily=trailing.fillna(k["cost_usd"]), level_source="real",
                    is_anomaly=0, anomaly_type="none", incident_id="")


def make_run(dataset: str, k: float, seed: int, startup_levels, kaggle_clean):
    rng = np.random.default_rng(seed)
    if dataset == "startup":
        clean = aug.build_normal(startup_levels, rng)
        df, truth = aug.inject_incidents(clean.copy(), (k, k), rng)
    else:
        clean = kaggle_clean
        df, truth = aug.inject_incidents(clean.copy(), (k, k), rng, KAGGLE_INCIDENTS, 6)
    W = df.pivot_table(index="date", columns="ServiceName", values="cost_usd",
                       aggfunc="sum", fill_value=0.0)
    clean_total = clean.groupby("date")["cost_usd"].sum().reindex(W.index)
    labels = df.groupby("date")["is_anomaly"].max().reindex(W.index).fillna(0).astype(int)
    return W, clean_total, labels, truth


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detection_flags(W: pd.DataFrame) -> pd.DataFrame:
    totals = W.sum(axis=1)
    flags = baseline_flags(totals, WARMUP)
    iso = detect(pd.DataFrame({"date": W.index, "total_cost_usd": totals.values}))
    flags.insert(0, "isolation_forest", iso["predicted"].values)
    flags.index = W.index
    return flags


def score_detection(flags, labels, extra, truth) -> list[dict]:
    ev = flags.index[WARMUP:]
    rows = []
    for det in DETECTORS:
        f, y = flags.loc[ev, det], labels.loc[ev]
        tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
        fn = int(((f == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        caught, delays, exposed, total_extra = 0, [], 0.0, 0.0
        for _, inc in truth.iterrows():
            window = pd.date_range(inc["start"], inc["end"], freq="D")
            hits = [d for d in window if flags.loc[d, det] == 1]
            ex = extra.loc[window]
            total_extra += ex.sum()
            if hits:
                caught += 1
                delays.append((hits[0] - window[0]).days)
                exposed += ex.loc[:hits[0] - pd.Timedelta(days=1)].sum()
            else:
                exposed += ex.sum()
        rows.append({"detector": det, "precision": p, "recall_days": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "event_recall": caught / len(truth),
                     "false_alerts_per_30d": fp / len(ev) * 30,
                     "mean_delay_days": np.mean(delays) if delays else np.nan,
                     "cost_exposed_pct": exposed / total_extra * 100 if total_extra else 0.0})
    return rows


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def score_attribution(W, truth, seed) -> list[dict]:
    days = [pd.Timestamp(s) for s in truth["start"]]
    ranks = rank_drivers(W, days, seed)
    rows = []
    for _, inc in truth.iterrows():
        day = pd.Timestamp(inc["start"])
        for m in METHODS:
            order = ranks[m][day]
            r = order.index(inc["driver_primary"]) + 1 if inc["driver_primary"] in order else np.inf
            row = {"method": m, "anomaly_type": inc["anomaly_type"], "rank": r,
                   "top1": r == 1, "top3": r <= 3, "rr": 0.0 if np.isinf(r) else 1 / r}
            if inc["driver_secondary"]:
                row["both_in_top3"] = {inc["driver_primary"], inc["driver_secondary"]} <= set(order[:3])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Real, unlabelled spikes
# --------------------------------------------------------------------------- #
def real_cases() -> str:
    lines = ["# Real cost spikes explained (no injected data)\n",
             "Driver ranking = cost_delta (increase vs each service's trailing 7-day median).\n"]
    k = pd.read_csv(CLEAN / "kaggle_daily_by_service.csv", parse_dates=["date"])
    W = k.pivot_table(index="date", columns="ServiceName", values="cost_usd", aggfunc="sum")
    t = W.sum(axis=1)
    flags = detection_flags(W)
    base = t.shift(1).rolling(7, min_periods=3).median()
    top = (t - base).iloc[WARMUP:].sort_values(ascending=False).head(5).index.sort_values()
    lines.append("## Public Azure subscription (Kaggle) - five largest day-over-normal increases\n")
    lines.append("| Date | Cost | Normal | Increase | Flagged by | Top drivers (increase) |")
    lines.append("|---|---|---|---|---|---|")
    for d in top:
        ex = explain_day(W, d, 3)
        drivers = "; ".join(f"{s} +{v:.2f}" for s, v in ex["increase"].items())
        by = ", ".join(c for c in DETECTORS if flags.loc[d, c]) or "none"
        lines.append(f"| {d.date()} | {t[d]:.2f} | {base[d]:.2f} | +{t[d]-base[d]:.2f} "
                     f"({(t[d]/base[d]-1)*100:.0f}%) | {by} | {drivers} |")

    s = pd.read_csv(CLEAN / "real_single_days.csv", parse_dates=["date"])
    S = s.pivot_table(index="ServiceName", columns="date", values="CostUSD", aggfunc="sum", fill_value=0)
    normal, spike = S[pd.Timestamp("2026-06-01")], S[pd.Timestamp("2026-06-28")]
    diff = (spike - normal).sort_values(ascending=False)
    # Service-level only (no resource names), used by the dashboard's Real cases page.
    pd.DataFrame({"normal_day": normal, "spike_day": spike, "change": spike - normal}).loc[diff.index] \
        .rename_axis("service").round(4).to_csv(OUT / "real_case_startup.csv")
    lines.append("\n## Case organisation (Sri Lankan startup) - 28 June 2026 vs 1 June 2026\n")
    lines.append(f"Total USD {spike.sum():.2f} vs {normal.sum():.2f} "
                 f"(+{(spike.sum()/normal.sum()-1)*100:.0f}%).\n")
    lines.append("| Service | 1 Jun | 28 Jun | Change |")
    lines.append("|---|---|---|---|")
    for svc, v in diff.items():
        if abs(v) >= 0.005:
            lines.append(f"| {svc} | {normal[svc]:.3f} | {spike[svc]:.3f} | {v:+.3f} |")
    lines.append("\nNote: fixed-price services are ~58% of their normal day on 28 June, so that "
                 "export likely covers part of the day; the AI-service increase is real regardless.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def fig_detection(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, facecolor=SURFACE)
    for col, ds in enumerate(["startup", "kaggle"]):
        for row, (metric, label) in enumerate([("event_recall", "Incidents detected (event recall)"),
                                               ("false_alerts_per_30d", "False alerts per 30 days")]):
            ax = axes[row, col]
            for det, colr in zip(DETECTORS, PALETTE):
                d = summary[(summary.dataset == ds) & (summary.detector == det)]
                ax.plot(d["k"] * 100, d[metric], color=colr, lw=2, marker="o", ms=5, label=det)
            name = "Case startup (anchored synthetic)" if ds == "startup" else "Public Azure (real daily + injected)"
            style(ax, f"{name}\n{label}")
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 30, 50, 100], ["5%", "10%", "20%", "30%", "50%", "100%"])
            if row == 1:
                ax.set_xlabel("Incident size (extra cost as % of a normal day)", color=INK2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_detection_vs_size.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_attribution(att: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    for ax, ds in zip(axes, ["startup", "kaggle"]):
        d = att[att.dataset == ds]
        width = 0.2
        for i, (m, colr) in enumerate(zip(METHODS, PALETTE)):
            sub = d[d.method == m].set_index("k").reindex(K_LEVELS)
            x = np.arange(len(K_LEVELS)) + (i - 1.5) * width
            ax.bar(x, sub["top1"] * 100, width=width - 0.02, color=colr, label=m)
        style(ax, "Case startup" if ds == "startup" else "Public Azure (real daily)")
        ax.set_xticks(np.arange(len(K_LEVELS)), [f"{int(k*100)}%" for k in K_LEVELS])
        ax.set_xlabel("Incident size", color=INK2)
    axes[0].set_ylabel("Top-1 driver accuracy (%)", color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_attribution.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, fmt: dict) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(fmt.get(c, "{}").format(v) for c, v in r.items()) + " |"
                     for r in df.to_dict("records"))
    return head + body


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, kclean = aug.monthly_levels(), kaggle_frame()
    det_rows, att_rows = [], []
    for ds in ["startup", "kaggle"]:
        for k in K_LEVELS:
            for seed in SEEDS:
                W, clean_total, labels, truth = make_run(ds, k, seed, levels, kclean)
                extra = (W.sum(axis=1) - clean_total).clip(lower=0)
                tag = {"dataset": ds, "k": k, "seed": seed}
                det_rows += [{**tag, **r} for r in score_detection(detection_flags(W), labels, extra, truth)]
                att_rows += [{**tag, **r} for r in score_attribution(W, truth, seed)]
            print(f"done {ds} k={k}")

    det = pd.DataFrame(det_rows)
    att = pd.DataFrame(att_rows)
    det.to_csv(OUT / "detection_runs.csv", index=False)
    att.to_csv(OUT / "attribution_runs.csv", index=False)

    dsum = det.groupby(["dataset", "k", "detector"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    asum = (att.groupby(["dataset", "k", "method"])
            .agg(top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean"),
                 n=("rr", "size")).reset_index())
    dsum.to_csv(OUT / "detection_summary.csv", index=False)
    asum.to_csv(OUT / "attribution_summary.csv", index=False)
    fig_detection(dsum)
    fig_attribution(asum)
    (OUT / "real_cases.md").write_text(real_cases())

    # ---- thesis report ----------------------------------------------------
    pct = "{:.0%}"
    rep = ["# Evaluation report\n",
           f"Runs: 2 datasets x {len(K_LEVELS)} incident sizes x {len(SEEDS)} seeds. "
           f"Evaluation excludes the first {WARMUP} warm-up days.\n"]
    for ds in ["startup", "kaggle"]:
        rep.append(f"\n## Detection - {ds} (incident size 30% of a normal day)\n")
        if ds == "kaggle":
            rep.append("The real background already contains unlabelled real spikes (see the real-cases "
                       "section), so alerts on them count as false here: precision is a lower bound.\n")
        t = dsum[(dsum.dataset == ds) & (dsum.k == 0.3)][
            ["detector", "precision", "recall_days", "f1", "event_recall",
             "false_alerts_per_30d", "mean_delay_days", "cost_exposed_pct"]]
        rep.append(md_table(t, {"precision": "{:.2f}", "recall_days": "{:.2f}", "f1": "{:.2f}",
                                "event_recall": pct, "false_alerts_per_30d": "{:.1f}",
                                "mean_delay_days": "{:.2f}", "cost_exposed_pct": "{:.0f}%"}))
        rep.append(f"\n### Event recall by incident size - {ds}\n")
        piv = dsum[dsum.dataset == ds].pivot(index="detector", columns="k", values="event_recall")
        piv.columns = [f"{int(c*100)}%" for c in piv.columns]
        rep.append(md_table(piv.reset_index(), {c: pct for c in piv.columns}))
        rep.append(f"\n## Attribution - {ds} (all sizes pooled)\n")
        a = att[att.dataset == ds].groupby("method").agg(
            top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean"),
            n=("rr", "size")).reindex(METHODS).reset_index()
        rep.append(md_table(a, {"top1": pct, "top3": pct, "mrr": "{:.2f}"}))
        rep.append(f"\n### Top-1 accuracy by incident size - {ds}\n")
        piv = asum[asum.dataset == ds].pivot(index="method", columns="k", values="top1").reindex(METHODS)
        piv.columns = [f"{int(c*100)}%" for c in piv.columns]
        rep.append(md_table(piv.reset_index(), {c: pct for c in piv.columns}))
        multi = att[(att.dataset == ds) & att.get("both_in_top3", pd.Series(dtype=object)).notna()]
        if not multi.empty:
            rep.append("\n### Multi-service incidents: both drivers in top-3\n")
            m = multi.groupby("method")["both_in_top3"].mean().reindex(METHODS).reset_index()
            rep.append(md_table(m, {"both_in_top3": pct}))
    rep.append("\n" + real_cases())
    (OUT / "REPORT.md").write_text("\n".join(rep))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
