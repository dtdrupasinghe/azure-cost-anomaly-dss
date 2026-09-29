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
  ml_evaluation.csv                              walk-forward + contamination-sensitivity runs
  fig_detection_ml.png, fig_detection_stat.png, fig_auc.png, fig_attribution.png
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
from detector import CONTAMINATION, ML_MODELS, all_flags, ml_features, ml_scores  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

CLEAN = ROOT / "Dataset" / "clean"
OUT = ROOT / "results" / "evaluation"
K_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
SEEDS = range(10)
WARMUP = 14
# consensus = final DSS detector (>= 2 of median_mad, moving_avg, three_sigma, isolation_forest)
STATS = ["median_mad", "moving_avg", "three_sigma", "ewma", "fixed_budget"]
DETECTORS = ["consensus"] + ML_MODELS + STATS
KIND = {"consensus": "Ensemble", **{m: "ML" for m in ML_MODELS}, **{m: "Statistical" for m in STATS}}
CONTAMINATIONS = [0.03, 0.05, 0.09, 0.15]

KAGGLE_INCIDENTS = [
    ("vm_left_running",     ["Virtual Machines"],                          (1, 3)),
    ("storage_growth",      ["Storage"],                                   (1, 2)),
    ("sql_scale_up",        ["SQL Database"],                              (1, 2)),
    ("log_ingestion_burst", ["Log Analytics"],                             (1, 1)),
    ("runaway_egress",      ["Bandwidth"],                                 (1, 2)),
    ("analytics_rollout",   ["Azure Synapse Analytics", "Log Analytics"],  (1, 2)),
]

# Reference palette, categorical slots 1-6 in fixed order (dataviz skill).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
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
def detection_flags(W: pd.DataFrame):
    """(flags, scores) for every detector on the daily total, fitted on the whole period."""
    return all_flags(W.sum(axis=1), with_scores=True)


def walk_forward(W: pd.DataFrame, seed: int, block: int = 7, contamination: float = CONTAMINATION):
    """Deployment-style ML evaluation: every `block` days, retrain each ML model on the PAST
    days only (expanding window) and score the next block. No future data is ever seen."""
    X, n = ml_features(W.sum(axis=1)), len(W)
    S = {m: np.zeros(n) for m in ML_MODELS}
    F = {m: np.zeros(n, dtype=int) for m in ML_MODELS}
    for b in range(WARMUP, n, block):
        rows = np.arange(b, min(b + block, n))
        s, f = ml_scores(X, np.arange(b), rows, contamination, seed)
        for m in ML_MODELS:
            S[m][rows], F[m][rows] = s[m], f[m]
    return pd.DataFrame(F, index=W.index), pd.DataFrame(S, index=W.index)


def score_detection(flags, labels, extra, truth, scores=None, detectors=DETECTORS) -> list[dict]:
    ev = flags.index[WARMUP:]
    rows = []
    for det in detectors:
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
        if scores is not None and y.nunique() == 2:   # threshold-free ranking quality
            rows[-1]["roc_auc"] = roc_auc_score(y, scores.loc[ev, det])
            rows[-1]["pr_auc"] = average_precision_score(y, scores.loc[ev, det])
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
    flags = detection_flags(W)[0]
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


def fig_detection(summary: pd.DataFrame, detectors: list[str], name: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, facecolor=SURFACE)
    for col, ds in enumerate(["startup", "kaggle"]):
        for row, (metric, label) in enumerate([("event_recall", "Incidents detected (event recall)"),
                                               ("false_alerts_per_30d", "False alerts per 30 days")]):
            ax = axes[row, col]
            for det, colr in zip(detectors, PALETTE):
                d = summary[(summary.dataset == ds) & (summary.detector == det)]
                ax.plot(d["k"] * 100, d[metric], color=colr, lw=2.6 if det == "consensus" else 2,
                        marker="o", ms=5, label=det)
            name_ds = "Case startup (anchored synthetic)" if ds == "startup" else "Public Azure (real daily + injected)"
            style(ax, f"{name_ds}\n{label}")
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 30, 50, 100], ["5%", "10%", "20%", "30%", "50%", "100%"])
            if row == 1:
                ax.set_xlabel("Incident size (extra cost as % of a normal day)", color=INK2)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=6,
               frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_auc(det: pd.DataFrame) -> None:
    """ROC-AUC per detector (all sizes and seeds), one bar per dataset."""
    m = det.groupby(["detector", "dataset"])["roc_auc"].mean().unstack()[["startup", "kaggle"]]
    m = m.reindex(m.mean(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    y = np.arange(len(m))
    for i, (ds, colr, lab) in enumerate([("startup", PALETTE[0], "Case startup"),
                                         ("kaggle", PALETTE[1], "Public Azure (real daily)")]):
        ax.barh(y + (i - 0.5) * 0.38, m[ds], height=0.34, color=colr, label=lab)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.505, len(m) - 0.4, "random ranking", color=INK2, fontsize=8)
    ax.set_yticks(y, [f"{d}  ({KIND[d]})" for d in m.index])
    ax.set_xlim(0.4, 1.0)
    style(ax, "How well each detector ranks anomalous days (ROC-AUC, higher is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc.png", dpi=150, facecolor=SURFACE)
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


def significance(det: pd.DataFrame, ref: str, others: list[str]) -> pd.DataFrame:
    """Wilcoxon signed-rank test on paired F1 (same dataset, size and seed) of `ref` vs each other."""
    rows = []
    for ds in ["startup", "kaggle"]:
        piv = det[det.dataset == ds].pivot_table(index=["k", "seed"], columns="detector", values="f1")
        for o in others:
            diff = piv[ref] - piv[o]
            try:
                p = wilcoxon(piv[ref], piv[o], zero_method="zsplit").pvalue
            except ValueError:
                p = 1.0
            rows.append({"dataset": ds, "comparison": f"{ref} vs {o}", "mean_f1_diff": diff.mean(),
                         "ref_better_runs": f"{(diff > 0).sum()}/{len(diff)}", "p_value": p})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    levels, kclean = aug.monthly_levels(), kaggle_frame()
    det_rows, att_rows, ml_rows = [], [], []
    for ds in ["startup", "kaggle"]:
        for k in K_LEVELS:
            for seed in SEEDS:
                W, clean_total, labels, truth = make_run(ds, k, seed, levels, kclean)
                extra = (W.sum(axis=1) - clean_total).clip(lower=0)
                tag = {"dataset": ds, "k": k, "seed": seed}
                flags, scores = detection_flags(W)
                det_rows += [{**tag, **r} for r in score_detection(flags, labels, extra, truth, scores)]
                att_rows += [{**tag, **r} for r in score_attribution(W, truth, seed)]
                if k == 0.3:   # ML-specific experiments at the reference incident size
                    Fw, Sw = walk_forward(W, seed)
                    ml_rows += [{**tag, "experiment": "walk_forward", "contamination": CONTAMINATION, **r}
                                for r in score_detection(Fw, labels, extra, truth, Sw, ML_MODELS)]
                    X, rows = ml_features(W.sum(axis=1)), np.arange(len(W))
                    for c in CONTAMINATIONS:
                        s_c, f_c = ml_scores(X, rows, rows, c, seed)
                        Fc = pd.DataFrame(f_c, index=W.index)
                        Sc = pd.DataFrame(s_c, index=W.index)
                        ml_rows += [{**tag, "experiment": "contamination", "contamination": c, **r}
                                    for r in score_detection(Fc, labels, extra, truth, Sc, ML_MODELS)]
            print(f"done {ds} k={k}")

    det, att, mlx = pd.DataFrame(det_rows), pd.DataFrame(att_rows), pd.DataFrame(ml_rows)
    det.to_csv(OUT / "detection_runs.csv", index=False)
    att.to_csv(OUT / "attribution_runs.csv", index=False)
    mlx.to_csv(OUT / "ml_evaluation.csv", index=False)

    dsum = det.groupby(["dataset", "k", "detector"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    asum = (att.groupby(["dataset", "k", "method"])
            .agg(top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean"),
                 n=("rr", "size")).reset_index())
    dsum.to_csv(OUT / "detection_summary.csv", index=False)
    asum.to_csv(OUT / "attribution_summary.csv", index=False)
    fig_detection(dsum, ["consensus"] + ML_MODELS, "fig_detection_ml.png",
                  "Machine-learning detectors vs the consensus detector")
    fig_detection(dsum, ["consensus"] + STATS, "fig_detection_stat.png",
                  "Statistical detectors vs the consensus detector")
    fig_auc(det)
    fig_attribution(asum)
    (OUT / "real_cases.md").write_text(real_cases())

    # ---- thesis report ----------------------------------------------------
    pct = "{:.0%}"
    f2 = "{:.2f}"
    rep = ["# Evaluation report\n",
           f"Runs: 2 datasets x {len(K_LEVELS)} incident sizes x {len(SEEDS)} seeds = "
           f"{2 * len(K_LEVELS) * len(SEEDS)} runs per detector. Evaluation excludes the first {WARMUP} "
           "warm-up days. Labels are used only for scoring, never for fitting; model settings were "
           "fixed in advance, not tuned on labels.\n",
           "\n## Models compared\n",
           "| Detector | Type | Rule / model |\n|---|---|---|",
           "| consensus | Ensemble (final DSS) | anomalous when >= 2 of isolation_forest, median_mad, moving_avg, three_sigma agree |",
           "| isolation_forest | ML | 200 trees, contamination 0.09, 4 features |",
           "| one_class_svm | ML | RBF kernel, nu 0.09, standardised features |",
           "| lof | ML | Local Outlier Factor, 10 neighbours, contamination 0.09 |",
           "| kmeans | ML | 3 clusters; distance to nearest centre, top 9% flagged |",
           "| median_mad | Statistical | > trailing-14-day median + 3.5 robust SD |",
           "| moving_avg | Statistical | > 1.2 x trailing 7-day mean |",
           "| three_sigma | Statistical | > trailing-14-day mean + 3 SD |",
           "| ewma | Statistical | > EWMA(span 7) mean + 3 EWMA SD |",
           "| fixed_budget | Statistical | > 1.3 x mean of the first 14 days |",
           "\nAll ML models use the same 4 features: daily total, deviation from the 7-day median, "
           "day of week, weekend flag.\n"]

    sel = dsum.groupby(["detector", "dataset"])["f1"].mean().unstack()[["startup", "kaggle"]]
    sel["both"] = sel.mean(axis=1)
    sel = sel.sort_values("both", ascending=False).reset_index()
    sel.insert(1, "type", sel["detector"].map(KIND))
    rep.append("\n## Detector selection (F1 averaged over all incident sizes)\n")
    rep.append("The DSS uses the detector with the best F1 across both datasets. The consensus rule was "
               "fixed before the extra ML models were added, so it was not tuned on these results.\n")
    rep.append(md_table(sel, {"startup": f2, "kaggle": f2, "both": f2}))

    # ML model evaluation --------------------------------------------------
    rep.append("\n## ML model evaluation\n")
    rep.append("### 1. Threshold-free ranking quality (ROC-AUC and PR-AUC, all sizes, mean ± SD over runs)\n")
    rep.append("ROC-AUC = chance that a random anomalous day scores higher than a random normal day "
               "(0.5 = random). PR-AUC focuses on the rare anomalous days.\n")
    auc = (det.groupby(["dataset", "detector"])
           .agg(roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"), pr=("pr_auc", "mean"),
                pr_sd=("pr_auc", "std")).reset_index())
    for ds in ["startup", "kaggle"]:
        t = auc[auc.dataset == ds].sort_values("roc", ascending=False)
        t = pd.DataFrame({"detector": t["detector"], "type": t["detector"].map(KIND),
                          "ROC-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.roc, t.roc_sd)],
                          "PR-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.pr, t.pr_sd)]})
        rep.append(f"\n**{ds}**\n")
        rep.append(md_table(t, {}))

    rep.append("\n### 2. ML models at incident size 30% (mean ± SD over 10 seeds)\n")
    for ds in ["startup", "kaggle"]:
        t = (det[(det.dataset == ds) & (det.k == 0.3) & det.detector.isin(ML_MODELS + ["consensus"])]
             .groupby("detector").agg(f1=("f1", "mean"), f1_sd=("f1", "std"), rec=("event_recall", "mean"),
                                      fa=("false_alerts_per_30d", "mean"), roc=("roc_auc", "mean"))
             .sort_values("f1", ascending=False).reset_index())
        t = pd.DataFrame({"detector": t.detector, "F1": [f"{a:.2f} ± {b:.2f}" for a, b in zip(t.f1, t.f1_sd)],
                          "incidents caught": t.rec.map(pct.format), "false alerts / 30d": t.fa.map(f"{{:.1f}}".format),
                          "ROC-AUC": t.roc.map("{:.3f}".format)})
        rep.append(f"\n**{ds}**\n")
        rep.append(md_table(t, {}))

    rep.append("\n### 3. Walk-forward (deployment-style) vs whole-period fitting, incident size 30%\n")
    rep.append("Walk-forward retrains each ML model every 7 days on past days only and scores the next "
               "7 days, as a live deployment would. Whole-period fitting sees all days at once.\n")
    batch = (det[(det.k == 0.3) & det.detector.isin(ML_MODELS)]
             .groupby(["dataset", "detector"])[["f1", "roc_auc"]].mean())
    wf = (mlx[mlx.experiment == "walk_forward"].groupby(["dataset", "detector"])[["f1", "roc_auc"]].mean())
    cmp_ = batch.join(wf, lsuffix="_whole", rsuffix="_walk").reset_index()
    rep.append(md_table(cmp_, {c: f2 for c in cmp_.columns if c not in ("dataset", "detector")}))

    rep.append("\n### 4. Sensitivity to the contamination setting (incident size 30%, F1)\n")
    rep.append("Contamination = expected share of anomalous days. 0.09 is the fixed prior used everywhere else.\n")
    sens = (mlx[mlx.experiment == "contamination"].groupby(["dataset", "detector", "contamination"])["f1"]
            .mean().unstack("contamination").reset_index())
    sens.columns = [c if isinstance(c, str) else f"c={c}" for c in sens.columns]
    rep.append(md_table(sens, {c: f2 for c in sens.columns if c.startswith("c=")}))

    rep.append("\n### 5. Statistical significance (Wilcoxon signed-rank test on paired F1, 60 runs per dataset)\n")
    rep.append("p < 0.05 means the difference is unlikely to be chance. `ref_better_runs` = runs where the "
               "first detector had the higher F1.\n")
    sig = pd.concat([significance(det, "consensus", [d for d in DETECTORS if d != "consensus"]),
                     significance(det, "isolation_forest", [m for m in ML_MODELS if m != "isolation_forest"])])
    rep.append(md_table(sig, {"mean_f1_diff": "{:+.3f}", "p_value": "{:.4f}"}))

    # Detection / attribution detail --------------------------------------
    for ds in ["startup", "kaggle"]:
        rep.append(f"\n## Detection - {ds} (incident size 30% of a normal day)\n")
        if ds == "kaggle":
            rep.append("The real background already contains unlabelled real spikes (see the real-cases "
                       "section), so alerts on them count as false here: precision is a lower bound.\n")
        t = dsum[(dsum.dataset == ds) & (dsum.k == 0.3)][
            ["detector", "precision", "recall_days", "f1", "event_recall",
             "false_alerts_per_30d", "mean_delay_days", "cost_exposed_pct", "roc_auc"]]
        rep.append(md_table(t, {"precision": f2, "recall_days": f2, "f1": f2, "roc_auc": f2,
                                "event_recall": pct, "false_alerts_per_30d": "{:.1f}",
                                "mean_delay_days": f2, "cost_exposed_pct": "{:.0f}%"}))
        rep.append(f"\n### Event recall by incident size - {ds}\n")
        piv = dsum[dsum.dataset == ds].pivot(index="detector", columns="k", values="event_recall")
        piv.columns = [f"{int(c*100)}%" for c in piv.columns]
        rep.append(md_table(piv.reset_index(), {c: pct for c in piv.columns}))
        rep.append(f"\n## Attribution - {ds} (all sizes pooled)\n")
        a = att[att.dataset == ds].groupby("method").agg(
            top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean"),
            n=("rr", "size")).reindex(METHODS).reset_index()
        rep.append(md_table(a, {"top1": pct, "top3": pct, "mrr": f2}))
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
