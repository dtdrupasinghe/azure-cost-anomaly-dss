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
from detector import CONSENSUS_MEMBERS, all_flags  # noqa: E402

CLEAN = ROOT / "Dataset" / "clean"
EVAL = ROOT / "results" / "evaluation"
DECISIONS = ROOT / "results" / "decision_log.csv"

DATASETS = {
    "Case startup": {
        "file": CLEAN / "augmented_daily.csv", "cur": "$", "truth": CLEAN / "ground_truth_incidents.csv",
        "short": "Case startup", "key": "startup",
        "note": "Daily series anchored on the case organisation's real Azure bills, with incidents of known cause."},
    "Public Azure": {
        "file": CLEAN / "kaggle_daily_by_service.csv", "cur": "", "truth": None,
        "short": "Public Azure subscription", "key": "kaggle",
        "note": "Real daily Azure billing (Kaggle, c.carrucciu 2023). Amounts in the billing currency."},
}
# The DSS runs one detector, chosen by the evaluation: the consensus of four detectors
# (anomalous when >= 2 agree). The others appear only on the Evaluation page.
DET = "consensus"
DETECTOR_NAMES = {"consensus": "Consensus (final)", "median_mad": "Median + MAD",
                  "moving_avg": "Moving-average band", "isolation_forest": "Isolation Forest",
                  "three_sigma": "3-sigma rule", "fixed_budget": "Fixed budget"}
METHOD_NAMES = {"cost_delta": "Cost change vs normal", "robust_z": "Relative change (robust z)",
                "shap_if": "SHAP on Isolation Forest", "largest_cost": "Largest cost that day"}
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
PAGES = {"Overview": ":material/space_dashboard:", "Investigate": ":material/troubleshoot:",
         "Evaluation": ":material/fact_check:", "Real cases": ":material/receipt_long:"}

ACCENT, RED, MUTED, GRID, INK2 = "#2a78d6", "#e34948", "#b8bcc6", "#eef0f3", "#5b5f6b"

st.set_page_config(page_title="Cost Anomaly DSS", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=block');
:root { --bg:#f5f6f8; --card:#ffffff; --line:#e7e9ee; --ink:#12141a; --ink2:#5b5f6b; --ink3:#8d919b;
  --accent:#2a78d6; --accent-soft:#eaf2fc; --red:#d23c3b; --red-soft:#fdeeee; --orange:#c9591c;
  --orange-soft:#fdf1e7; --amber:#9a6b00; --amber-soft:#fcf5e1; --green:#1f8a4c; --green-soft:#e7f5ec; }
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); }
.stApp, .stApp p, .stApp div, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp li,
.stApp span:not([data-testid="stIconMaterial"]):not(.msr) { font-family: 'Inter', sans-serif; }
.msr { font-family: 'Material Symbols Rounded' !important; font-size: 20px; line-height: 1; font-weight: normal;
  font-style: normal; display: inline-block; -webkit-font-smoothing: antialiased; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebarContent"] { padding-top: 0.6rem; }
.brand { display:flex; align-items:center; gap:12px; padding: 4px 2px 18px 2px; border-bottom:1px solid var(--line); margin-bottom: 14px; }
.logo { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, #2a78d6 0%, #5b9df0 100%); color:#fff; box-shadow: 0 2px 6px rgba(42,120,214,.35); }
.bname { font-weight:700; font-size:0.98rem; color:var(--ink); letter-spacing:-0.01em; }
.bsub { font-size:0.75rem; color:var(--ink3); margin-top:1px; }
.navlabel { font-size:0.68rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--ink3); margin: 16px 0 6px 2px; }
section[data-testid="stSidebar"] .stElementContainer:has([data-testid="stRadio"]),
section[data-testid="stSidebar"] [data-testid="stRadio"], section[data-testid="stSidebar"] [data-testid="stRadio"] > div { width: 100% !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; width: 100%; display: flex; flex-direction: column; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { padding: 9px 12px; border-radius: 9px; width: 100% !important; max-width: none; margin: 0; transition: background .15s; box-sizing: border-box; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #f2f4f7; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: var(--accent-soft); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: var(--accent); font-weight: 600; }
section[data-testid="stSidebar"] div[role="radiogroup"] p { font-size: 0.92rem; color: #2b2e36; }
.modelbox { font-size:0.8rem; color:var(--ink2); line-height:1.5; background:#f6f7f9; border:1px solid var(--line);
  border-radius:10px; padding:10px 12px; margin-bottom: 12px; }
.modelbox b { color:var(--ink); font-weight:600; }
.sidefoot { font-size:0.74rem; color:var(--ink3); line-height:1.45; margin-top: 22px; padding-top: 14px; border-top:1px solid var(--line); }

/* Page header */
.page-head { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; margin-bottom: 18px; }
.page-head > div:first-child { flex: 1 1 auto; min-width: 0; }
.crumb { font-size:0.74rem; font-weight:600; letter-spacing:0.07em; text-transform:uppercase; color:var(--ink3); }
.ptitle { font-size:1.65rem; font-weight:700; letter-spacing:-0.02em; color:var(--ink); margin-top:4px; line-height:1.2; }
.psub { font-size:0.9rem; color:var(--ink2); margin-top:4px; }
.chip { display:inline-flex; align-items:center; gap:8px; background:#fff; border:1px solid var(--line); border-radius:999px;
  padding:7px 14px; font-size:0.82rem; color:var(--ink2); font-weight:500; white-space:nowrap; }
.live { width:8px; height:8px; border-radius:50%; background:#1faa5a; box-shadow:0 0 0 3px rgba(31,170,90,.18); }

/* Cards */
[class*="st-key-card"] { background: var(--card); border:1px solid var(--line); border-radius:14px; padding: 18px 20px 14px 20px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.ctitle { font-size:0.98rem; font-weight:600; color:var(--ink); }
.csub { font-size:0.8rem; color:var(--ink3); margin-top:2px; margin-bottom: 6px; }
.chead { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.chead > div:first-child { flex: 1 1 auto; min-width: 0; } .chead .pill { flex: none; white-space: nowrap; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,.04); height:100%; }
.kpi-top { display:flex; align-items:center; gap:10px; }
.kpi-icon { width:32px; height:32px; border-radius:9px; display:flex; align-items:center; justify-content:center; }
.kpi-icon .msr { font-size:19px; }
.kpi-label { font-size:0.8rem; color:var(--ink2); font-weight:500; }
.kpi-value { font-size:1.55rem; font-weight:700; color:var(--ink); margin-top:12px; letter-spacing:-0.02em; }
.kpi-value.small { font-size:1.1rem; line-height:1.35; margin-top:14px; }
.kpi-sub { font-size:0.78rem; color:var(--ink3); margin-top:2px; }
.legend { display:flex; gap:16px; font-size:0.78rem; color:var(--ink2); align-items:center; flex-wrap: wrap; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:0; }
.dash { display:inline-block; width:14px; border-top:2px dashed #b8bcc6; margin-right:6px; vertical-align:3px; }

/* Pills and lists */
.pill { display:inline-flex; align-items:center; gap:5px; padding:3px 9px; border-radius:999px; font-size:0.72rem; font-weight:600; letter-spacing:0.01em; }
.pill.high { background:var(--red-soft); color:var(--red); } .pill.medium { background:var(--orange-soft); color:var(--orange); }
.pill.low { background:var(--amber-soft); color:var(--amber); } .pill.ok { background:var(--green-soft); color:var(--green); }
.pill.neutral { background:#f1f2f5; color:var(--ink2); }
.pill .msr { font-size:14px; }
.item { display:flex; justify-content:space-between; align-items:center; padding:11px 0; border-bottom:1px solid var(--line); }
.item:last-child { border-bottom:none; }
.i-date { font-size:0.86rem; font-weight:600; color:var(--ink); }
.i-sub { font-size:0.78rem; color:var(--ink3); margin-top:2px; }
.i-right { text-align:right; }
.i-amt { font-size:0.86rem; font-weight:600; color:var(--ink); margin-top:4px; }
.callout { display:flex; gap:12px; background:var(--accent-soft); border-radius:12px; padding:14px 16px; font-size:0.9rem; color:#1d3f6b; line-height:1.5; margin: 6px 0 14px 0; }
.callout .msr { color:var(--accent); }
.step { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--line); font-size:0.88rem; line-height:1.45; }
.step:last-child { border-bottom:none; }
.num { flex:none; width:24px; height:24px; border-radius:50%; background:#f1f2f5; color:var(--ink2); font-size:0.75rem; font-weight:600;
  display:flex; align-items:center; justify-content:center; }
.step b { color:var(--ink); } .step span { color:var(--ink2); }
.banner { display:flex; justify-content:space-between; align-items:center; gap:20px; flex-wrap:wrap; background:#fff; border:1px solid var(--line);
  border-radius:14px; padding:18px 22px; box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom: 16px; }
.b-title { font-size:1.05rem; font-weight:700; color:var(--ink); margin-top:6px; }
.stats { display:flex; gap:34px; flex-wrap:wrap; }
.stat-l { font-size:0.74rem; color:var(--ink3); font-weight:500; }
.stat-v { font-size:1.08rem; font-weight:700; color:var(--ink); margin-top:3px; }
.bar-row { display:grid; grid-template-columns: 170px 1fr 70px; gap:10px; align-items:center; padding:5px 0; font-size:0.84rem; }
.bar-track { background:#f1f2f5; border-radius:6px; height:8px; overflow:hidden; }
.bar-fill { background:var(--red); height:8px; border-radius:6px; }
.bar-name { color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-val { text-align:right; color:var(--ink2); font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
div[data-testid="stExpander"] details { border:1px solid var(--line); border-radius:12px; background:#fff; }
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
    flags = all_flags(total)
    normal = total.shift(1).rolling(7, min_periods=3).median()
    out = pd.DataFrame({"total": total, "normal": normal}).join(flags)
    out["extra"] = (out["total"] - out["normal"]).clip(lower=0).fillna(0)
    out["pct"] = (out["extra"] / out["normal"]).fillna(0)
    return out


@st.cache_data
def shap_for(W: pd.DataFrame, day) -> pd.Series:
    return shap_contributions(W, [day]).iloc[0]


@st.cache_data
def top_driver(W: pd.DataFrame, day) -> str:
    return explain_day(W, day, 1).index[0]


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
    lines = [f"*Azure cost anomaly on {day.date()}* ({severity(row['pct'])[0]} severity)",
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
def html(s: str, where=st) -> None:
    where.markdown(s, unsafe_allow_html=True)


def severity(pct: float) -> tuple[str, str]:
    """(label, css class) from the increase over normal."""
    if pct >= 0.75:
        return "High", "high"
    if pct >= 0.25:
        return "Medium", "medium"
    return "Low", "low"


def pill(pct: float) -> str:
    label, cls = severity(pct)
    return f'<span class="pill {cls}">{label}</span>'


def page_head(crumb: str, title: str, sub: str, chip: str) -> None:
    html(f'<div class="page-head"><div><div class="crumb">{crumb}</div><div class="ptitle">{title}</div>'
         f'<div class="psub">{sub}</div></div><div class="chip"><span class="live"></span>{chip}</div></div>')


def kpi(col, icon: str, tint: str, label: str, value: str, sub: str = "", small: bool = False) -> None:
    soft = {"#2a78d6": "#eaf2fc", "#d23c3b": "#fdeeee", "#1f8a4c": "#e7f5ec", "#c9591c": "#fdf1e7",
            "#7b5cd6": "#f1edfc"}[tint]
    html(f'<div class="kpi"><div class="kpi-top"><span class="kpi-icon" style="background:{soft};color:{tint}">'
         f'<span class="msr">{icon}</span></span><span class="kpi-label">{label}</span></div>'
         f'<div class="kpi-value{" small" if small else ""}">{value}</div><div class="kpi-sub">{sub or "&nbsp;"}</div></div>', col)


def card_head(title: str, sub: str = "", right: str = "") -> None:
    html(f'<div class="chead"><div><div class="ctitle">{title}</div><div class="csub">{sub}</div></div>{right}</div>')


def styled(chart, height: int):
    return (chart.properties(height=height).configure(font="Inter")
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, domain=False, ticks=False, labelColor=INK2,
                            titleColor=INK2, labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
                            labelPadding=8)
            .configure_axisX(grid=False))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
html('<div class="brand"><div class="logo"><span class="msr">query_stats</span></div><div>'
     '<div class="bname">Cost Anomaly DSS</div><div class="bsub">Azure cost decision support</div></div></div>',
     st.sidebar)
page = st.sidebar.radio("Navigation", [f"{icon}  {name}" for name, icon in PAGES.items()],
                        label_visibility="collapsed").split("  ", 1)[1]

html('<div class="navlabel">Data source</div>', st.sidebar)
ds_name = st.sidebar.selectbox("Dataset", list(DATASETS) + ["Upload a CSV"], label_visibility="collapsed")
html('<div class="navlabel">Detection model</div><div class="modelbox"><b>Consensus detector</b><br>'
     'A day is flagged when at least 2 of 4 detectors agree: Isolation Forest, median + MAD, '
     'moving-average band and 3-sigma. Chosen for the best F1 across both evaluation datasets.</div>',
     st.sidebar)
det, det_label = DET, "Consensus detector"
with st.sidebar.expander("Alert channel"):
    platform = st.selectbox("Platform", ["Slack", "Microsoft Teams"])
    webhook = st.text_input("Incoming webhook URL", type="password", placeholder="https://hooks...")
html('<div class="sidefoot">Recommends checks only. It never changes Azure resources.<br>'
     'BSc MIS research prototype · NSBM Green University</div>', st.sidebar)

if ds_name in DATASETS:
    cfg = DATASETS[ds_name]
    W = load(str(cfg["file"]))
else:
    cfg = {"cur": "", "truth": None, "short": "Uploaded data", "key": "upload", "note": "Uploaded daily per-service costs."}
    up = st.file_uploader("Daily per-service costs (CSV)", type=["csv"])
    if up is None:
        page_head("Data source", "Upload a cost file",
                  "A CSV with a date column (date, UsageDate or Date), a service column (ServiceName or "
                  "MeterCategory) and a cost column (cost_usd, CostUSD or CostInBillingCurrency).", "Waiting for data")
        st.stop()
    W = to_wide(pd.read_csv(up))

cur = cfg["cur"]
def money(v: float) -> str:  # noqa: E302
    return f"{cur}{v:,.2f}"
num_fmt = f"{cur}%.2f"

A = analyse(W)
flagged = A[A[det] == 1]
span = f"{A.index.min():%d %b %Y} – {A.index.max():%d %b %Y}"
chip = f"Monitoring · {det_label}"

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    page_head("Overview", cfg["short"], f"{cfg['note']} {span}.", chip)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "calendar_month", "#2a78d6", "Days monitored", f"{len(A)}", span)
    kpi(k2, "payments", "#7b5cd6", "Total spend", money(A["total"].sum()), f"{money(A['total'].mean())} per day on average")
    kpi(k3, "warning", "#d23c3b", "Anomalous days", f"{len(flagged)}",
        f"{(flagged['pct'] >= 0.75).sum()} high · {((flagged['pct'] >= 0.25) & (flagged['pct'] < 0.75)).sum()} medium severity")
    kpi(k4, "trending_up", "#c9591c", "Extra spend", money(flagged["extra"].sum()),
        f"{flagged['extra'].sum() / A['total'].sum():.1%} of total spend")
    st.write("")

    left, right = st.columns([2.15, 1], gap="medium")
    with left.container(key="card_trend"):
        card_head("Daily spend", "Hover for details. Normal = median of the previous 7 days.",
                  f'<div class="legend"><span><span class="dot" style="background:{ACCENT}"></span>Daily cost</span>'
                  f'<span><span class="dash"></span>Normal</span>'
                  f'<span><span class="dot" style="background:{RED}"></span>Anomaly</span></div>')
        plot = A.reset_index()[["date", "total", "normal"]].copy()
        plot["anomaly"] = A[det].values == 1
        base = alt.Chart(plot).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)))
        grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color="rgba(42,120,214,0)", offset=0),
                                   alt.GradientStop(color="rgba(42,120,214,0.20)", offset=1)])
        area = base.mark_area(color=grad, line={"color": ACCENT, "strokeWidth": 2}, interpolate="monotone").encode(
            y=alt.Y("total:Q", title=None))
        normal = base.mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(y="normal:Q")
        pts = base.transform_filter("datum.anomaly").mark_circle(
            color=RED, size=64, opacity=1, stroke="white", strokeWidth=2).encode(y="total:Q")
        hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
        rule = base.mark_rule(color="#c5c9d2", strokeWidth=1).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("total:Q", title="Cost", format=",.2f"),
                     alt.Tooltip("normal:Q", title="Normal", format=",.2f")]).add_params(hover)
        st.altair_chart(styled(area + normal + pts + rule, 372), use_container_width=True)

    with right.container(key="card_recent"):
        card_head("Recent anomalies", f"Latest {min(5, len(flagged))} of {len(flagged)}")
        if flagged.empty:
            html('<div class="csub">No anomalous days with the current method.</div>')
        else:
            items = "".join(
                f'<div class="item"><div><div class="i-date">{d:%d %b %Y}</div>'
                f'<div class="i-sub">{top_driver(W, d)}</div></div>'
                f'<div class="i-right">{pill(r["pct"])}<div class="i-amt">+{money(r["extra"])}</div></div></div>'
                for d, r in flagged.iloc[::-1].head(5).iterrows())
            html(items)

    st.write("")
    with st.container(key="card_table"):
        card_head("All anomalous days", "Main driver = service with the largest increase over its normal cost.")
        if not flagged.empty:
            tbl = pd.DataFrame({
                "Date": flagged.index.date,
                "Severity": [severity(p)[0] for p in flagged["pct"]],
                "Cost": flagged["total"].values, "Normal": flagged["normal"].values,
                "Extra cost": flagged["extra"].values, "Increase": (flagged["pct"] * 100).values,
                "Main driver": [top_driver(W, d) for d in flagged.index],
                "Detectors agreeing": flagged["votes"].values,
            }).iloc[::-1]
            st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                "Date": st.column_config.DateColumn(format="D MMM YYYY"),
                "Cost": st.column_config.NumberColumn(format=num_fmt),
                "Normal": st.column_config.NumberColumn(format=num_fmt),
                "Extra cost": st.column_config.NumberColumn(format=num_fmt),
                "Increase": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                            max_value=float(max(100, tbl["Increase"].max()))),
                "Detectors agreeing": st.column_config.ProgressColumn(format="%d of 4", min_value=0, max_value=4),
            })

# --------------------------------------------------------------------------- #
# Investigate
# --------------------------------------------------------------------------- #
elif page == "Investigate":
    page_head("Investigate", "Explain and decide", f"{cfg['short']} · {span}", chip)
    if flagged.empty:
        html('<div class="csub">No anomalous days with the current method.</div>')
        st.stop()
    pick, _ = st.columns([1, 2])
    day = pick.selectbox("Anomalous day", list(flagged.index[::-1]),
                         format_func=lambda d: f"{d:%d %b %Y}  ·  {money(A.loc[d, 'total'])}  ·  "
                                               f"{severity(A.loc[d, 'pct'])[0]}")
    r = A.loc[day]
    drivers = explain_day(W, day, 8)
    top = drivers[drivers["increase"] > 0].head(3)

    html(f'<div class="banner"><div>{pill(r["pct"])}<div class="b-title">Cost anomaly on {day:%A, %d %B %Y}</div>'
         f'<div class="csub">Flagged by {int(r["votes"])} of 4 detectors</div></div>'
         f'<div class="stats"><div><div class="stat-l">Cost that day</div><div class="stat-v">{money(r["total"])}</div></div>'
         f'<div><div class="stat-l">Normal</div><div class="stat-v">{money(r["normal"])}</div></div>'
         f'<div><div class="stat-l">Extra cost</div><div class="stat-v">+{money(r["extra"])}</div></div>'
         f'<div><div class="stat-l">Increase</div><div class="stat-v">{r["pct"]:+.0%}</div></div>'
         f'<div><div class="stat-l">If it continues 30 days</div><div class="stat-v">{money(r["extra"] * 30)}</div></div>'
         f'</div></div>')

    left, right = st.columns([1.35, 1], gap="medium")
    with left.container(key="card_drivers"):
        card_head("Cost drivers", "Change in each service's cost compared with its normal level")
        bars = drivers[drivers["increase"].abs() > 0.005 * drivers["increase"].abs().max()]
        bars = bars.rename_axis("service").reset_index()
        bar = alt.Chart(bars).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("increase:Q", title=None),
            y=alt.Y("service:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
            color=alt.condition("datum.increase > 0", alt.value(RED), alt.value(MUTED)),
            tooltip=[alt.Tooltip("service", title="Service"), alt.Tooltip("cost:Q", format=",.2f", title="Cost"),
                     alt.Tooltip("normal:Q", format=",.2f", title="Normal"),
                     alt.Tooltip("increase:Q", format="+,.2f", title="Change"),
                     alt.Tooltip("share_of_increase:Q", format=".0%", title="Share of increase")])
        st.altair_chart(styled(bar, 34 * len(bars)).configure_axisY(grid=False), use_container_width=True)
        with st.expander("Model view: SHAP contributions"):
            sv = shap_for(W, day).sort_values().head(8).rename_axis("service").rename("shap").reset_index()
            html('<div class="csub">Isolation Forest trained on per-service changes. More negative values pushed '
                 'the day further towards anomalous. In the evaluation, the cost-change ranking named the true '
                 'driver more often than SHAP.</div>')
            st.altair_chart(styled(alt.Chart(sv).mark_bar(color=ACCENT, cornerRadiusEnd=4, height=14).encode(
                x=alt.X("shap:Q", title=None), y=alt.Y("service:N", sort="x", title=None),
                tooltip=[alt.Tooltip("service", title="Service"), alt.Tooltip("shap:Q", format=".3f", title="SHAP")]),
                30 * len(sv)).configure_axisY(grid=False), use_container_width=True)

    with right.container(key="card_why"):
        card_head("Why it was flagged")
        if not top.empty:
            lead = top.iloc[0]
            new = " It was not in use before, which points to a new deployment." if lead["normal"] < 1e-6 else ""
            html(f'<div class="callout"><span class="msr">lightbulb</span><div><b>{top.index[0]}</b> accounts for '
                 f'<b>{lead["share_of_increase"]:.0%}</b> of the increase: +{money(lead["increase"])} over its '
                 f'normal {money(lead["normal"])}.{new}</div></div>')
        card_head("Suggested checks", "Ordered by contribution to the increase")
        html("".join(f'<div class="step"><div class="num">{i}</div><div><b>{svc}</b><br>'
                     f'<span>{ACTIONS.get(svc, DEFAULT_ACTION)}</span></div></div>'
                     for i, svc in enumerate(top.index, 1)))

    st.write("")
    with st.container(key="card_decision"):
        card_head("Decision", "Record how this anomaly is handled. Decisions are kept in a log for follow-up.")
        with st.form("decision", border=False):
            c1, c2 = st.columns([1.2, 1])
            choice = c1.segmented_control("Classification", ["Expected change", "Investigate", "Resolved", "False alarm"],
                                          default="Investigate")
            note = c2.text_input("Note", placeholder="Optional")
            b1, b2, _ = st.columns([1, 1, 3])
            save = b1.form_submit_button("Record decision", type="primary", use_container_width=True)
            send = b2.form_submit_button("Send alert", use_container_width=True)
        if save:
            rec = pd.DataFrame([{"recorded_at": datetime.now().isoformat(timespec="seconds"),
                                 "dataset": cfg["short"], "date": day.date(), "cost": round(r["total"], 4),
                                 "extra_cost": round(r["extra"], 4), "severity": severity(r["pct"])[0],
                                 "top_driver": top.index[0] if not top.empty else "",
                                 "detector": det, "decision": choice or "", "note": note}])
            rec.to_csv(DECISIONS, mode="a", header=not DECISIONS.exists(), index=False)
            st.success("Decision recorded.")
        if send:
            if webhook:
                ok, info = send_alert(webhook, platform, alert_text(day, r, drivers, money))
                st.success(f"Alert sent ({info}).") if ok else st.error(f"Alert failed: {info}")
            else:
                st.warning("Add a webhook URL under Alert channel in the sidebar.")
        if DECISIONS.exists():
            log = pd.read_csv(DECISIONS)
            html(f'<div class="csub" style="margin-top:14px">Decision log · {len(log)} entries</div>')
            st.dataframe(log.iloc[::-1].head(10), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
elif page == "Evaluation":
    page_head("Evaluation", "How well does it work?",
              "Controlled tests with incidents of known cause: 2 datasets × 6 incident sizes × 10 random seeds", chip)
    dsum = pd.read_csv(EVAL / "detection_summary.csv") if (EVAL / "detection_summary.csv").exists() else None
    asum = pd.read_csv(EVAL / "attribution_summary.csv") if (EVAL / "attribution_summary.csv").exists() else None

    if cfg.get("truth") and Path(cfg["truth"]).exists():
        truth = pd.read_csv(cfg["truth"], parse_dates=["start", "end"])
        res = pd.DataFrame([{
            "Incident": inc["incident_id"], "Start": inc["start"].date(),
            "Type": inc["anomaly_type"].replace("_", " ").capitalize(), "True driver": inc["driver_primary"],
            "Detected": "Yes" if any(A.loc[d, det] == 1 for d in pd.date_range(inc["start"], inc["end"])) else "No",
            "Driver named": top_driver(W, inc["start"]),
            "Extra cost": inc["extra_cost_usd"]} for _, inc in truth.iterrows()])
        res["Driver correct"] = (res["Driver named"] == res["True driver"]).map({True: "Yes", False: "No"})
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "radar", "#2a78d6", "Incidents detected", f"{(res['Detected'] == 'Yes').sum()} of {len(res)}", det_label)
        kpi(k2, "target", "#1f8a4c", "Driver named correctly", f"{(res['Driver correct'] == 'Yes').sum()} of {len(res)}",
            "First-ranked service")
        if dsum is not None:
            both = dsum.groupby("detector")["f1"].mean()
            kpi(k3, "insights", "#7b5cd6", "Detection model in use", "Consensus (2 of 4)",
                f"F1 {both['consensus']:.2f} averaged over both datasets, best of 6", small=True)
        if asum is not None:
            pooled = asum[asum.dataset == "kaggle"].groupby("method")["top1"].mean()
            kpi(k4, "psychology", "#c9591c", "Best explanation method", METHOD_NAMES[pooled.idxmax()],
                f"{pooled.max():.0%} top-1 accuracy on real data", small=True)
        st.write("")
        with st.container(key="card_incidents"):
            card_head("Injected incidents in this dataset", "Known cause, so detection and explanation can be scored")
            st.dataframe(res, hide_index=True, width="stretch", height=35 * (len(res) + 1) + 3, column_config={
                "Start": st.column_config.DateColumn(format="D MMM YYYY"),
                "Extra cost": st.column_config.NumberColumn(format="$%.2f")})
        st.write("")

    if dsum is not None:
        with st.container(key="card_detectors"):
            card_head("Detection models compared", "Why the consensus detector is used. Incidents 30% above a "
                      "normal day, ranked by F1")
            cols = st.columns(2, gap="medium")
            for col, (ds, label) in zip(cols, [("startup", "Case startup"), ("kaggle", "Public Azure (real daily)")]):
                t = dsum[(dsum.dataset == ds) & (dsum.k == 0.3)].sort_values("f1", ascending=False)
                col.markdown(f"**{label}**")
                col.dataframe(pd.DataFrame({
                    "Method": t["detector"].map(DETECTOR_NAMES), "F1": t["f1"],
                    "Incidents caught": t["event_recall"] * 100,
                    "False alerts / 30 days": t["false_alerts_per_30d"], "Cost exposed": t["cost_exposed_pct"]}),
                    hide_index=True, width="stretch", column_config={
                        "F1": st.column_config.NumberColumn(format="%.2f", help="Balances catches against false alerts"),
                        "Incidents caught": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                        "False alerts / 30 days": st.column_config.NumberColumn(format="%.1f"),
                        "Cost exposed": st.column_config.NumberColumn(format="%.0f%%",
                                                                      help="Share of incident cost before the first alert")})
        st.write("")
    if asum is not None:
        with st.container(key="card_methods"):
            card_head("Explanation methods compared", "Share of incidents where the true driver was ranked first "
                      "or in the top three (all incident sizes)")
            cols = st.columns(2, gap="medium")
            for col, (ds, label) in zip(cols, [("startup", "Case startup"), ("kaggle", "Public Azure (real daily)")]):
                t = asum[asum.dataset == ds].groupby("method")[["top1", "top3", "mrr"]].mean() \
                    .sort_values("top1", ascending=False)
                col.markdown(f"**{label}**")
                col.dataframe(pd.DataFrame({"Method": t.index.map(METHOD_NAMES), "Top-1": t["top1"].values * 100,
                                            "Top-3": t["top3"].values * 100, "MRR": t["mrr"].values}),
                              hide_index=True, width="stretch", column_config={
                                  "Top-1": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                  "Top-3": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                  "MRR": st.column_config.NumberColumn(format="%.2f")})
        st.write("")
    with st.expander("Figures by incident size"):
        for img in ("fig_detection_vs_size.png", "fig_attribution.png"):
            if (EVAL / img).exists():
                st.image(str(EVAL / img), width="stretch")
    if dsum is None:
        st.warning("Run `python scripts/evaluate.py` to generate the evaluation results.")

# --------------------------------------------------------------------------- #
# Real cases
# --------------------------------------------------------------------------- #
else:
    page_head("Real cases", "Real cost spikes, explained",
              "No injected data: actual Azure billing from the case organisation and a public subscription", chip)

    def bars_html(items: pd.Series) -> str:
        m = max(items.max(), 1e-9)
        return "".join(f'<div class="bar-row"><div class="bar-name">{s}</div><div class="bar-track">'
                       f'<div class="bar-fill" style="width:{v / m * 100:.0f}%"></div></div>'
                       f'<div class="bar-val">+{v:,.2f}</div></div>' for s, v in items.items())

    case_file = EVAL / "real_case_startup.csv"
    if case_file.exists():
        c = pd.read_csv(case_file, index_col="service")
        before, after = c["normal_day"].sum(), c["spike_day"].sum()
        with st.container(key="card_case_startup"):
            card_head("Case organisation · 28 June 2026",
                      f"Compared with 1 June 2026, a normal day. Total ${after:.2f} vs ${before:.2f}.",
                      pill(after / before - 1).replace("</span>", f" · {after / before - 1:+.0%}</span>"))
            html(bars_html(c["change"][c["change"] > 0.005]))
            html('<div class="csub" style="margin-top:10px">A new AI agent (search grounding and GPT tokens) drove '
                 'the spike. Fixed-price services are about 58% of a normal day, so the export likely covers part '
                 'of the day; the AI-service increase is real regardless.</div>')
        st.write("")

    Wk = load(str(DATASETS["Public Azure"]["file"]))
    Ak = analyse(Wk)
    inc = (Ak["total"] - Ak["normal"]).iloc[14:].sort_values(ascending=False).head(4).index.sort_values()
    html('<div class="ctitle" style="margin:4px 0 10px 2px">Public Azure subscription · largest increases over normal</div>')
    for i, d in enumerate(inc):
        if i % 2 == 0:
            cols = st.columns(2, gap="medium")
        rr = Ak.loc[d]
        ex = explain_day(Wk, d, 3)
        with cols[i % 2].container(key=f"card_case_k{i}"):
            card_head(f"{d:%d %B %Y}", f"Cost {rr['total']:,.2f} vs normal {rr['normal']:,.2f} "
                      f"· {'flagged' if rr[DET] else 'not flagged'} ({int(rr['votes'])} of 4 detectors)",
                      pill(rr["pct"]).replace("</span>", f" · {rr['pct']:+.0%}</span>"))
            html(bars_html(ex["increase"].clip(lower=0)))
        if i % 2 == 1:
            st.write("")
