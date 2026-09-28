"""Explainable Azure cost-anomaly Decision Support System — Streamlit dashboard.

Workflow shown to the user (thesis main RQ: identify -> explain -> manage):
  1. Identify  : daily Azure cost, a normal-cost baseline, and flagged anomalous days
  2. Explain   : which services drove the increase (cost_delta attribution) + SHAP view
  3. Impact    : extra cost of each anomaly and its monthly exposure if it persists
  4. Manage    : suggested checks per service, a human decision that is logged,
                 and Slack / Microsoft Teams alerts

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "scripts"))
from attribution import explain_day, shap_contributions  # noqa: E402
from detector import baseline_flags, detect  # noqa: E402

CLEAN = ROOT / "Dataset" / "clean"
EVAL = ROOT / "results" / "evaluation"
DECISIONS = ROOT / "results" / "decision_log.csv"

DATASETS = {
    "Case startup (Jan–Jun 2026)": {
        "file": CLEAN / "augmented_daily.csv", "money": "${:,.2f}",
        "truth": CLEAN / "ground_truth_incidents.csv",
        "note": "Daily series anchored on the case organisation's real Azure bills, with incidents of known cause."},
    "Public Azure subscription (Kaggle)": {
        "file": CLEAN / "kaggle_daily_by_service.csv", "money": "{:,.2f}", "truth": None,
        "note": "Real daily Azure billing, Dec 2022 – Mar 2023. Amounts in the subscription's billing currency."},
}
DETECTORS = {
    "Median + MAD (recommended)": "median_mad",
    "Moving-average band": "moving_avg",
    "Isolation Forest": "isolation_forest",
    "3-sigma rule": "three_sigma",
    "Fixed budget": "fixed_budget",
}
# Suggested first checks per service (decision support, not automated action).
ACTIONS = {
    "Virtual Machines": "Look for VMs left running, resized or newly created. Deallocate idle VMs and enable auto-shutdown.",
    "Storage": "Look for new disks, snapshots or data growth. Review Premium disks and lifecycle rules.",
    "Bandwidth": "Check outbound data transfer: large downloads, cross-region replication, CDN configuration.",
    "Foundry Models": "Check AI model token usage. Set quotas or rate limits and make sure no agent is looping.",
    "MS Bing Services": "Check search calls made by AI agents. Cap usage or disable it if unintended.",
    "SQL Database": "Check for a tier or scale change (DTU/vCore) and long-running queries.",
    "Log Analytics": "Check log ingestion volume. Reduce verbose logging and set a daily ingestion cap.",
    "Azure App Service": "Check for a new App Service plan or scale-out. Downscale unused plans.",
    "Azure Firewall": "A new firewall carries a fixed hourly charge. Confirm it was planned.",
    "Azure Synapse Analytics": "Check for pools left running or heavy queries. Pause dedicated SQL pools.",
}
DEFAULT_ACTION = "Review recent changes to this service in the Azure Activity Log and confirm they were intended."

ACCENT, MUTED, RED, GRID, INK2 = "#2a78d6", "#b5b4ae", "#e34948", "#eeede9", "#6b6a65"

st.set_page_config(page_title="Cost Anomaly DSS", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, select, textarea { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
h1 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; margin-top: 1.6rem !important; }
section[data-testid="stSidebar"] { background: #fafaf9; border-right: 1px solid #eeede9; }
.overline { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #6b6a65; margin-bottom: 0.25rem; }
.subtle { color: #6b6a65; font-size: 0.88rem; margin: 0.25rem 0 1.4rem 0; }
.kpi { border: 1px solid #eeede9; border-radius: 12px; padding: 16px 18px; background: #fff; height: 100%; }
.kpi-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: #6b6a65; }
.kpi-value { font-size: 1.6rem; font-weight: 600; color: #1a1a19; margin-top: 6px; letter-spacing: -0.01em; }
.kpi-sub { font-size: 0.8rem; color: #8a8984; margin-top: 2px; }
.callout { border-left: 3px solid #2a78d6; background: #f5f8fd; padding: 14px 18px; border-radius: 0 10px 10px 0; font-size: 0.95rem; margin: 0.4rem 0 1rem 0; }
.action { display: flex; gap: 16px; padding: 12px 0; border-bottom: 1px solid #eeede9; font-size: 0.92rem; }
.action b { min-width: 190px; font-weight: 600; }
.action span { color: #52514e; }
.legend { font-size: 0.8rem; color: #6b6a65; display: flex; gap: 18px; margin-top: -4px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.dash { display: inline-block; width: 16px; border-top: 2px dashed #b5b4ae; margin-right: 6px; vertical-align: 3px; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid #eeede9; }
.stTabs [data-baseweb="tab"] { padding: 10px 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data + analysis
# --------------------------------------------------------------------------- #
def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Any per-service daily file -> wide matrix (date x service)."""
    date = next(c for c in ("date", "UsageDate", "Date") if c in df.columns)
    svc = next(c for c in ("ServiceName", "MeterCategory") if c in df.columns)
    cost = next(c for c in ("cost_usd", "CostUSD", "CostInBillingCurrency", "Cost") if c in df.columns)
    df = df.assign(_d=pd.to_datetime(df[date], errors="coerce")).dropna(subset=["_d"])
    W = df.pivot_table(index="_d", columns=svc, values=cost, aggfunc="sum", fill_value=0.0)
    W.index.name = "date"
    return W.asfreq("D", fill_value=0.0)


@st.cache_data
def load(path: str) -> pd.DataFrame:
    return to_wide(pd.read_csv(path))


@st.cache_data
def analyse(W: pd.DataFrame) -> pd.DataFrame:
    total = W.sum(axis=1)
    flags = baseline_flags(total)
    iso = detect(pd.DataFrame({"date": W.index, "total_cost_usd": total.values}))
    flags.insert(0, "isolation_forest", iso["predicted"].values)
    flags.index = W.index
    normal = total.shift(1).rolling(7, min_periods=3).median()
    out = pd.DataFrame({"total": total, "normal": normal}).join(flags)
    out["extra"] = (out["total"] - out["normal"]).clip(lower=0).fillna(0)
    return out


@st.cache_data
def shap_for(W: pd.DataFrame, day) -> pd.Series:
    return shap_contributions(W, [day]).iloc[0]


def send_alert(webhook: str, platform: str, message: str) -> tuple[bool, str]:
    """Post a message to a Slack or Microsoft Teams incoming webhook."""
    try:
        if platform == "Microsoft Teams":
            payload = {"@type": "MessageCard", "@context": "http://schema.org/extensions",
                       "themeColor": "E34948", "summary": "Azure cost anomaly", "text": message}
        else:
            payload = {"text": message}
        r = requests.post(webhook, json=payload, timeout=10)
        return (200 <= r.status_code < 300), f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def alert_text(day, row, drivers: pd.DataFrame, money) -> str:
    lines = [f"*Azure cost anomaly on {day.date()}*",
             f"Cost {money(row['total'])} vs normal {money(row['normal'])} "
             f"(+{money(row['extra'])}, about {money(row['extra'] * 30)} per month if it continues)",
             "Top drivers:"]
    for svc, r in drivers.head(3).iterrows():
        if r["increase"] > 0:
            lines.append(f"- {svc}: +{money(r['increase'])} ({r['share_of_increase']:.0%} of the increase)")
    lines.append("_Sent by the Azure Cost Anomaly DSS._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub or "&nbsp;"}</div></div>',
                 unsafe_allow_html=True)


def styled(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height)
            .configure(font="Inter")
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, domain=False, ticks=False,
                            labelColor=INK2, titleColor=INK2, labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal", labelPadding=8)
            .configure_axisX(grid=False))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown('<div class="overline">Data</div>', unsafe_allow_html=True)
ds_name = st.sidebar.selectbox("Dataset", list(DATASETS) + ["Upload a CSV"], label_visibility="collapsed")
if ds_name in DATASETS:
    cfg = DATASETS[ds_name]
    W = load(str(cfg["file"]))
else:
    cfg = {"money": "{:,.2f}", "truth": None, "note": "Uploaded daily per-service costs."}
    up = st.sidebar.file_uploader("Daily per-service costs", type=["csv"])
    if up is None:
        st.markdown('<div class="overline">Azure cost anomaly decision support</div>', unsafe_allow_html=True)
        st.title("Upload a cost file")
        st.markdown('<p class="subtle">A CSV with a date column (date, UsageDate or Date), a service column '
                    '(ServiceName or MeterCategory) and a cost column (cost_usd, CostUSD or '
                    'CostInBillingCurrency).</p>', unsafe_allow_html=True)
        st.stop()
    W = to_wide(pd.read_csv(up))
money = cfg["money"].format

st.sidebar.markdown('<div class="overline" style="margin-top:1.4rem">Detection method</div>', unsafe_allow_html=True)
det_label = st.sidebar.selectbox("Detection method", list(DETECTORS), label_visibility="collapsed")
det = DETECTORS[det_label]

st.sidebar.markdown('<div class="overline" style="margin-top:1.4rem">Alerts</div>', unsafe_allow_html=True)
platform = st.sidebar.selectbox("Platform", ["Slack", "Microsoft Teams"])
webhook = st.sidebar.text_input("Incoming webhook URL", type="password", placeholder="https://hooks...")

A = analyse(W)
flagged = A[A[det] == 1]

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.markdown('<div class="overline">Azure cost anomaly decision support</div>', unsafe_allow_html=True)
st.title(ds_name)
st.markdown(f'<p class="subtle">{cfg["note"]} Recommends checks only; it never changes Azure resources.</p>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Days monitored", f"{len(A)}", f"{A.index.min():%d %b %Y} – {A.index.max():%d %b %Y}")
kpi(k2, "Total spend", money(A["total"].sum()), f"{money(A['total'].mean())} per day on average")
kpi(k3, "Anomalous days", f"{len(flagged)}", det_label.replace(" (recommended)", ""))
kpi(k4, "Extra spend", money(flagged["extra"].sum()),
    f"{flagged['extra'].sum() / A['total'].sum():.1%} of total spend")
st.write("")

tab_mon, tab_exp, tab_eval, tab_real = st.tabs(["Overview", "Explain & decide", "Evaluation", "Real cases"])

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tab_mon:
    plot = A.reset_index()[["date", "total", "normal"]].copy()
    plot["anomaly"] = A[det].values == 1
    base = alt.Chart(plot).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")))
    line = base.mark_line(color=ACCENT, strokeWidth=2).encode(
        y=alt.Y("total:Q", title="Daily cost"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("total:Q", format=",.2f", title="Cost"),
                 alt.Tooltip("normal:Q", format=",.2f", title="Normal")])
    normal = base.mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(y="normal:Q")
    pts = base.transform_filter("datum.anomaly").mark_circle(
        color=RED, size=70, opacity=1, stroke="white", strokeWidth=2).encode(
        y="total:Q", tooltip=[alt.Tooltip("date:T", title="Date"),
                              alt.Tooltip("total:Q", format=",.2f", title="Cost")])
    st.altair_chart(styled(normal + line + pts, 340).interactive(), use_container_width=True)
    st.markdown(f'<div class="legend"><span><span class="dot" style="background:{ACCENT}"></span>Daily cost</span>'
                f'<span><span class="dash"></span>Normal (median of previous 7 days)</span>'
                f'<span><span class="dot" style="background:{RED}"></span>Flagged day</span></div>',
                unsafe_allow_html=True)

    st.subheader(f"Flagged days · {len(flagged)}")
    if flagged.empty:
        st.markdown('<p class="subtle">No anomalous days with the current method.</p>', unsafe_allow_html=True)
    else:
        rows = []
        for day, r in flagged.iterrows():
            top = explain_day(W, day, 1)
            agree = sum(A.loc[day, d] for d in DETECTORS.values())
            rows.append({"Date": day.date(), "Cost": money(r["total"]), "Normal": money(r["normal"]),
                         "Extra cost": money(r["extra"]), "Main driver": top.index[0],
                         "Methods agreeing": f"{agree} of 5"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 2. Explain & decide
# --------------------------------------------------------------------------- #
with tab_exp:
    if flagged.empty:
        st.markdown('<p class="subtle">No flagged days to explain.</p>', unsafe_allow_html=True)
    else:
        day = st.selectbox("Flagged day", list(flagged.index[::-1]),
                           format_func=lambda d: f"{d:%d %b %Y}  ·  {money(A.loc[d, 'total'])}")
        r = A.loc[day]
        drivers = explain_day(W, day, 8)
        top = drivers[drivers["increase"] > 0].head(3)

        c1, c2, c3 = st.columns(3)
        kpi(c1, "Cost that day", money(r["total"]), f"Normal {money(r['normal'])}")
        kpi(c2, "Increase", f"+{money(r['extra'])}",
            f"{r['total'] / r['normal'] - 1:+.0%} vs normal" if r["normal"] else "")
        kpi(c3, "Monthly exposure", money(r["extra"] * 30), "If the increase continues for 30 days")

        st.subheader("Why this day was flagged")
        if not top.empty:
            lead = top.iloc[0]
            new = " It was not in use before, which points to a new deployment." if lead["normal"] < 1e-6 else ""
            st.markdown(f'<div class="callout"><b>{top.index[0]}</b> accounts for '
                        f'<b>{lead["share_of_increase"]:.0%}</b> of the increase: '
                        f'+{money(lead["increase"])} over its normal {money(lead["normal"])}.{new}</div>',
                        unsafe_allow_html=True)
        bars = drivers.rename_axis("service").reset_index()
        st.altair_chart(styled(
            alt.Chart(bars).mark_bar(cornerRadiusEnd=4, height=16).encode(
                x=alt.X("increase:Q", title="Change vs normal"),
                y=alt.Y("service:N", sort="-x", title=None),
                color=alt.condition("datum.increase > 0", alt.value(RED), alt.value(MUTED)),
                tooltip=[alt.Tooltip("service", title="Service"),
                         alt.Tooltip("cost:Q", format=",.2f", title="Cost"),
                         alt.Tooltip("normal:Q", format=",.2f", title="Normal"),
                         alt.Tooltip("increase:Q", format="+,.2f", title="Change"),
                         alt.Tooltip("share_of_increase:Q", format=".0%", title="Share of increase")]),
            34 * len(bars)).configure_axisY(grid=False), use_container_width=True)

        with st.expander("Model view: SHAP contributions"):
            sv = shap_for(W, day).sort_values().head(8).rename_axis("service").rename("shap").reset_index()
            st.markdown('<p class="subtle">Isolation Forest trained on per-service changes. More negative values '
                        'pushed the day further towards anomalous. In the evaluation, the cost-change ranking '
                        'above named the true driver more often than SHAP.</p>', unsafe_allow_html=True)
            st.altair_chart(styled(alt.Chart(sv).mark_bar(color=ACCENT, cornerRadiusEnd=4, height=14).encode(
                x=alt.X("shap:Q", title="SHAP value"), y=alt.Y("service:N", sort="x", title=None),
                tooltip=[alt.Tooltip("service", title="Service"), alt.Tooltip("shap:Q", format=".3f", title="SHAP")]),
                30 * len(sv)).configure_axisY(grid=False), use_container_width=True)

        st.subheader("Suggested checks")
        st.markdown("".join(f'<div class="action"><b>{svc}</b><span>{ACTIONS.get(svc, DEFAULT_ACTION)}</span></div>'
                            for svc in top.index), unsafe_allow_html=True)

        st.subheader("Decision")
        with st.form("decision", border=False):
            choice = st.radio("Classify this anomaly", ["Expected change", "Investigate", "Resolved", "False alarm"],
                              horizontal=True)
            note = st.text_input("Note", placeholder="Optional")
            b1, b2, _ = st.columns([1, 1, 3])
            save = b1.form_submit_button("Record decision", type="primary", use_container_width=True)
            send = b2.form_submit_button("Send alert", use_container_width=True)
        if save:
            rec = pd.DataFrame([{"recorded_at": datetime.now().isoformat(timespec="seconds"),
                                 "dataset": ds_name, "date": day.date(), "cost": round(r["total"], 4),
                                 "extra_cost": round(r["extra"], 4),
                                 "top_driver": top.index[0] if not top.empty else "",
                                 "detector": det, "decision": choice, "note": note}])
            rec.to_csv(DECISIONS, mode="a", header=not DECISIONS.exists(), index=False)
            st.success("Decision recorded.")
        if send:
            if webhook:
                ok, info = send_alert(webhook, platform, alert_text(day, r, drivers, money))
                st.success(f"Alert sent ({info}).") if ok else st.error(f"Alert failed: {info}")
            else:
                st.warning("Add a webhook URL in the sidebar first.")
        if DECISIONS.exists():
            log = pd.read_csv(DECISIONS)
            st.markdown(f'<p class="subtle" style="margin:1rem 0 0.4rem">Decision log · {len(log)} entries</p>',
                        unsafe_allow_html=True)
            st.dataframe(log.tail(10), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 3. Evaluation
# --------------------------------------------------------------------------- #
with tab_eval:
    if cfg.get("truth") and Path(cfg["truth"]).exists():
        truth = pd.read_csv(cfg["truth"], parse_dates=["start", "end"])
        rows = []
        for _, inc in truth.iterrows():
            hit = any(A.loc[d, det] == 1 for d in pd.date_range(inc["start"], inc["end"]))
            named = explain_day(W, inc["start"], 1).index[0]
            rows.append({"Incident": inc["incident_id"], "Start": inc["start"].date(),
                         "Type": inc["anomaly_type"].replace("_", " "), "True driver": inc["driver_primary"],
                         "Detected": "Yes" if hit else "No", "Driver named": named,
                         "Driver correct": "Yes" if named == inc["driver_primary"] else "No"})
        res = pd.DataFrame(rows)
        c1, c2, _ = st.columns([1, 1, 2])
        kpi(c1, "Incidents detected", f"{(res['Detected'] == 'Yes').sum()} of {len(res)}",
            det_label.replace(" (recommended)", ""))
        kpi(c2, "Driver named correctly", f"{(res['Driver correct'] == 'Yes').sum()} of {len(res)}",
            "First-ranked service")
        st.subheader("Injected incidents with known cause")
        st.dataframe(res, width="stretch", hide_index=True)
    else:
        st.markdown('<p class="subtle">This dataset has real spikes but no injected ground truth. See the '
                    'controlled evaluation below and the Real cases tab.</p>', unsafe_allow_html=True)
    st.subheader("Controlled evaluation")
    st.markdown('<p class="subtle">Two datasets, six incident sizes, ten random seeds each.</p>',
                unsafe_allow_html=True)
    for img in ("fig_detection_vs_size.png", "fig_attribution.png"):
        if (EVAL / img).exists():
            st.image(str(EVAL / img), width="stretch")
    if not (EVAL / "fig_detection_vs_size.png").exists():
        st.warning("Run `python scripts/evaluate.py` to generate the evaluation results.")

# --------------------------------------------------------------------------- #
# 4. Real cases
# --------------------------------------------------------------------------- #
with tab_real:
    if (EVAL / "real_cases.md").exists():
        st.markdown((EVAL / "real_cases.md").read_text())
    else:
        st.warning("Run `python scripts/evaluate.py` to generate the real-case analysis.")
