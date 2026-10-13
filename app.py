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
        "note": "Real billing, with known incidents"},
    "Public Azure": {
        "file": CLEAN / "kaggle_daily_by_service.csv", "cur": "", "truth": None,
        "short": "Public Azure subscription", "key": "kaggle",
        "note": "Real daily billing (Kaggle, 2023)"},
}
# The DSS runs one detector, chosen by the evaluation: the consensus of four detectors
# (anomalous when >= 2 agree). The others appear only on the Model Evaluation page.
DET = "consensus"
DETECTOR_NAMES = {"consensus": "Consensus (final)", "isolation_forest": "Isolation Forest (ML)",
                  "one_class_svm": "One-Class SVM (ML)", "lof": "Local Outlier Factor (ML)",
                  "kmeans": "K-Means (ML)", "median_mad": "Median + MAD", "moving_avg": "Moving-average band",
                  "three_sigma": "3-sigma rule", "ewma": "EWMA", "fixed_budget": "Fixed budget"}
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

# Grouped sidebar navigation: (section label, [(page name, material icon), ...]).
NAV_GROUPS = [
    ("OVERVIEW", [("Overview", ":material/space_dashboard:")]),
    ("ANALYSIS", [("Investigate", ":material/troubleshoot:"), ("Model Evaluation", ":material/fact_check:"),
                 ("Real Cases", ":material/receipt_long:")]),
    ("DATA", [("Data Source", ":material/database:")]),
    ("SYSTEM", [("Alerts", ":material/notifications:"), ("Settings", ":material/settings:")]),
]
# Decision-form choices, mapped to the status vocabulary shown on the Overview table.
STATUS_MAP = {"Expected change": "Expected", "Investigate": "Under Review",
             "Resolved": "Resolved", "False alarm": "False Alarm"}

# One red, everywhere: previously the chart used #e34948, the pills used #d23c3b
# and the bar chart used var(--red) -- three near-identical reds a viewer reads
# as "almost the same colour" (a Gestalt *similarity* failure: same meaning,
# should be one colour, not three close ones people can't quite place).
# Colour is reserved for meaning (severity, the one accent) -- not decoration:
# KPI icons and cards are flat/neutral rather than each tinted a different hue.
ACCENT, RED, MUTED, GRID, INK2 = "#2876d2", "#ce302f", "#b8bcc6", "#eef0f3", "#5b5f6b"

st.set_page_config(page_title="Cost Anomaly DSS", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=block');
:root {
  /* --- Color: 60/30/10 rule (bg 60%, card surface 30%, accent 10%). Every text
     pair below is WCAG-AA verified (>=4.5:1 for small/bold text, computed, not
     eyeballed) -- ink3 and the four status colors were adjusted from their first
     pass after that check failed. */
  --bg:#f5f6f8; --card:#ffffff; --line:#e7e9ee;
  --ink:#12141a; --ink2:#5b5f6b; --ink3:#6d717c;                 /* ink3: 2.92:1 -> 4.51:1 fixed */
  --accent:#2876d2; --accent-soft:#eaf2fc;                        /* darkened 2a78d6->2876d2: white-on-accent 4.42:1 -> 4.55:1 */
  --red:#ce302f; --red-soft:#fdeeee;                              /* 4.19:1 -> 4.56:1 */
  --orange:#b55019; --orange-soft:#fdf1e7;                        /* 3.84:1 -> 4.58:1 */
  --amber:#956700; --amber-soft:#fcf5e1;                          /* 4.31:1 -> 4.57:1 */
  --green:#1c7e45; --green-soft:#e7f5ec;                          /* 3.89:1 -> 4.53:1 */

  /* --- Type scale: geometric, ratio 1.25 (major third), base 14px. Replaces the
     previous ad hoc sizes, whose steps (0.68/0.8/0.9/1.55rem) were NOT a
     consistent ratio -- that inconsistency is what made the hierarchy feel
     slightly arbitrary. Every text size below is one of these seven steps. */
  --text-2xs: 11px;  /* eyebrow / overline labels (tracked, uppercase) */
  --text-xs:  12px;  /* captions, table meta, kpi labels */
  --text-sm:  14px;  /* body base */
  --text-base:16px;  /* emphasized body, card titles */
  --text-md:  18px;  /* not used standalone; kept for future card titles */
  --text-lg:  22px;  /* banner hero stat */
  --text-xl:  28px;  /* kpi value */
  --text-2xl: 35px;  /* page title -- was too close in size to the kpi value (26.4 vs 24.8px); now a clear step above it */

  /* --- Spacing: 8pt grid for every component-internal gap/padding. */
  --sp-1:4px; --sp-2:8px; --sp-3:16px; --sp-4:24px; --sp-5:32px; --sp-6:40px;
  /* --- Rhythm: Fibonacci steps (8/13/21/34), the standard way a golden-ratio
     (phi=1.618) relationship is expressed as a *spacing scale* rather than
     forced into a single fixed ratio -- used for vertical rhythm between
     stacked elements, where a plain doubling reads too mechanically. */
  --rh-1:8px; --rh-2:13px; --rh-3:21px; --rh-4:34px;

  --radius-sm:8px; --radius-md:12px; --radius-lg:16px; --radius-pill:999px;
}
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); }
.stApp, .stApp p, .stApp div, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp li,
.stApp span:not([data-testid="stIconMaterial"]):not(.msr) { font-family: 'Inter', sans-serif; }
.msr { font-family: 'Material Symbols Rounded' !important; font-size: 20px; line-height: 1; font-weight: normal;
  font-style: normal; display: inline-block; -webkit-font-smoothing: antialiased; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: var(--rh-4); padding-bottom: 48px; max-width: 1280px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebarContent"] { padding-top: var(--sp-2); }
.brand { display:flex; align-items:center; gap:var(--sp-3); padding: var(--sp-1); margin-bottom: var(--rh-2); }
.logo { width:38px; height:38px; border-radius:var(--radius-md); display:flex; align-items:center; justify-content:center;
  background: var(--accent); color:#fff; }
.bname { font-weight:700; font-size:var(--text-base); color:var(--ink); letter-spacing:-0.01em; }
.bsub { font-size:var(--text-xs); color:var(--ink3); margin-top:1px; }
.navlabel { font-size:var(--text-2xs); font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--ink3);
  margin: var(--rh-3) 0 var(--sp-1) var(--sp-1); }
/* Nav buttons styled as a flat list, not bordered buttons -- the active page is
   highlighted by a dynamically-injected rule below (the page is known in Python
   at render time, so the highlight is computed, not CSS-guessed). */
section[data-testid="stSidebar"] [class*="st-key-nav_"] button {
  width:100%; justify-content:flex-start; text-align:left; background:transparent; border:none;
  color:#2b2e36; font-weight:500; padding:var(--sp-2) var(--sp-3); border-radius:var(--radius-md);
  min-height:40px; box-shadow:none; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover { background:#f2f4f7; border:none; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button p { font-size:var(--text-base); }
section[data-testid="stSidebar"] [class*="st-key-nav_"] { margin-bottom:2px; }
.modelbox { font-size:var(--text-xs); color:var(--ink2); line-height:1.55; background:#f6f7f9; border:1px solid var(--line);
  border-radius:var(--radius-md); padding:var(--sp-2) var(--sp-3); margin-bottom: var(--sp-3); }
.modelbox b { color:var(--ink); font-weight:600; }
.sidefoot { font-size:var(--text-2xs); color:var(--ink3); line-height:1.5; margin-top: var(--rh-4); padding-top: var(--sp-3);
  border-top:1px solid var(--line); }

/* Page header -- eyebrow now sits on a white chip-free zone check: verified 4.51:1 on both bg and card */
.page-head { display:flex; justify-content:space-between; align-items:flex-end; gap:var(--sp-4); margin-bottom: var(--rh-3); }
.page-head > div:first-child { flex: 1 1 auto; min-width: 0; }
.crumb { font-size:var(--text-2xs); font-weight:600; letter-spacing:0.07em; text-transform:uppercase; color:var(--ink3); }
.ptitle { font-size:var(--text-2xl); font-weight:700; letter-spacing:-0.02em; color:var(--ink); margin-top:var(--sp-1); line-height:1.15; }
.psub { font-size:var(--text-sm); color:var(--ink2); margin-top:var(--sp-1); }
.chip { display:inline-flex; align-items:center; gap:var(--sp-2); background:#fff; border:1px solid var(--line); border-radius:var(--radius-pill);
  padding:7px var(--sp-3); font-size:var(--text-xs); color:var(--ink2); font-weight:500; white-space:nowrap; }
.live { width:8px; height:8px; border-radius:50%; background:#1c9350; }

/* Cards -- flat: a single hairline border carries the surface, no shadow. A
   shadow implies elevation/depth, which this flat, content-first layout
   doesn't need -- one less visual cue competing for attention. Interior
   padding ratio 16:26 (~1:1.618, golden ratio) rather than a plain square/
   equal padding, so cards read slightly wider than tall inside, echoing the
   same proportion as a printed page margin. */
[class*="st-key-card"] { background: var(--card); border:1px solid var(--line); border-radius:var(--radius-lg);
  padding: var(--sp-3) 26px var(--sp-2) 26px; }
.ctitle { font-size:var(--text-base); font-weight:600; color:var(--ink); }
.csub { font-size:var(--text-xs); color:var(--ink3); margin-top:var(--sp-1); margin-bottom: var(--sp-2); }
.chead { display:flex; justify-content:space-between; align-items:flex-start; gap:var(--sp-3); }
.chead > div:first-child { flex: 1 1 auto; min-width: 0; } .chead .pill { flex: none; white-space: nowrap; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:var(--radius-lg); padding:var(--sp-3) 26px; height:100%; }
.kpi-top { display:flex; align-items:center; gap:var(--sp-2); }
/* Icon is a plain neutral glyph, not a coloured chip -- a different tint per
   KPI card is decoration, not information (the label already says what the
   number is); colour is spent only where it carries meaning: severity. */
.kpi-icon { display:flex; align-items:center; justify-content:center; color:var(--ink3); }
.kpi-icon .msr { font-size:18px; }
.kpi-label { font-size:var(--text-xs); color:var(--ink2); font-weight:500; }
.kpi-value { font-size:var(--text-xl); font-weight:700; color:var(--ink); margin-top:var(--rh-1); letter-spacing:-0.02em; }
.kpi-value.small { font-size:var(--text-base); font-weight:600; line-height:1.35; margin-top:var(--rh-2); }
.kpi-sub { font-size:var(--text-xs); color:var(--ink3); margin-top:var(--sp-1); }
.legend { display:flex; gap:var(--sp-3); font-size:var(--text-xs); color:var(--ink2); align-items:center; flex-wrap: wrap; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:var(--sp-1); vertical-align:0; }
.dash { display:inline-block; width:14px; border-top:2px dashed #b8bcc6; margin-right:var(--sp-1); vertical-align:3px; }

/* Pills, tags and lists -- a severity badge used to be a solid colour block
   (background + coloured text); the label text already names the severity
   ("High"/"Medium"), so the colour was pure redundant decoration. A small dot
   carries the colour instead, text stays neutral -- same information, one
   visual element quieter. */
.pill { display:inline-flex; align-items:center; gap:6px; padding:2px 0;
  font-size:var(--text-2xs); font-weight:600; letter-spacing:0.02em; color:var(--ink2); }
.pill-dot { width:7px; height:7px; border-radius:50%; flex:none; }
.pill.high .pill-dot { background:var(--red); } .pill.medium .pill-dot { background:var(--orange); }
.pill.low .pill-dot { background:var(--amber); } .pill.ok .pill-dot { background:var(--green); }
.pill.neutral .pill-dot { background:var(--ink3); }
/* Tag = a neutral accent label for a fact, not a severity (e.g. "Real case") --
   deliberately a different shape (solid chip) from .pill so the two are never
   confused at a glance. */
.tag { display:inline-flex; align-items:center; font-size:var(--text-2xs); font-weight:700; letter-spacing:0.04em;
  text-transform:uppercase; color:var(--accent); background:var(--accent-soft); border-radius:var(--radius-sm);
  padding:3px 8px; }
.item { display:flex; justify-content:space-between; align-items:center; padding:var(--sp-3) 0; border-bottom:1px solid var(--line); }
.item:last-child { border-bottom:none; }
.i-date { font-size:var(--text-sm); font-weight:600; color:var(--ink); }
.i-sub { font-size:var(--text-xs); color:var(--ink3); margin-top:var(--sp-1); }
.i-right { text-align:right; }
.i-amt { font-size:var(--text-sm); font-weight:600; color:var(--ink); margin-top:var(--sp-1); }
.callout { display:flex; gap:var(--sp-3); background:var(--accent-soft); border-radius:var(--radius-md); padding:var(--sp-3);
  font-size:var(--text-base); color:#1d3f6b; line-height:1.5; margin: var(--sp-1) 0 var(--sp-3) 0; }
.callout .msr { color:var(--accent); }
.step { display:flex; gap:var(--sp-3); padding:var(--sp-2) 0; border-bottom:1px solid var(--line); font-size:var(--text-sm); line-height:1.45; }
.step:last-child { border-bottom:none; }
.num { flex:none; width:24px; height:24px; border-radius:50%; background:#f1f2f5; color:var(--ink2); font-size:var(--text-xs); font-weight:600;
  display:flex; align-items:center; justify-content:center; }
.step b { color:var(--ink); } .step span { color:var(--ink2); }
.banner { display:flex; justify-content:space-between; align-items:center; gap:var(--sp-5); flex-wrap:wrap; background:#fff;
  border:1px solid var(--line); border-radius:var(--radius-lg); padding:var(--sp-3) 26px;
  margin-bottom: var(--rh-3); }
.b-title { font-size:var(--text-lg); font-weight:700; color:var(--ink); margin-top:var(--rh-1); }
.stats { display:flex; gap:var(--rh-4); flex-wrap:wrap; }
.stat-l { font-size:var(--text-xs); color:var(--ink3); font-weight:500; }
.stat-v { font-size:var(--text-base); font-weight:700; color:var(--ink); margin-top:var(--sp-1); }
.bar-row { display:grid; grid-template-columns: 170px 1fr 70px; gap:var(--sp-2); align-items:center; padding:5px 0; font-size:var(--text-sm); }
.bar-track { background:#f1f2f5; border-radius:6px; height:8px; overflow:hidden; }
.bar-fill { background:var(--red); height:8px; border-radius:6px; }
.bar-name { color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-val { text-align:right; color:var(--ink2); font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] { gap: var(--sp-4); }
div[data-testid="stExpander"] details { border:1px solid var(--line); border-radius:var(--radius-md); background:#fff; }
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


def read_any(file) -> pd.DataFrame:
    """CSV or Excel, by extension -- so a bulk upload can mix either."""
    return pd.read_excel(file) if file.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(file)


def combine_wide(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine several wide matrices (e.g. one per monthly export) into one, aligning on date
    and service and summing any overlap, rather than requiring every file to share a schema."""
    services = sorted(set().union(*(f.columns for f in frames)))
    combined = frames[0].reindex(columns=services, fill_value=0.0)
    for f in frames[1:]:
        combined = combined.add(f.reindex(columns=services, fill_value=0.0), fill_value=0.0)
    return combined.sort_index().asfreq("D", fill_value=0.0)


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
    # A zero baseline (e.g. a short upload with little warmup) would otherwise divide
    # to +inf, which crashes table rendering (JSON has no infinity). Treat a jump
    # from zero as a capped 200% rather than an unrepresentable one.
    out["pct"] = (out["extra"] / out["normal"].where(out["normal"] > 0)).fillna(0)
    out.loc[(out["normal"] <= 0) & (out["extra"] > 0), "pct"] = 2.0
    return out


@st.cache_data
def shap_for(W: pd.DataFrame, day) -> pd.Series:
    return shap_contributions(W, [day]).iloc[0]


@st.cache_data
def top_driver(W: pd.DataFrame, day) -> str:
    return explain_day(W, day, 1).index[0]


def decision_status(dataset_short: str, day) -> str:
    """Latest recorded decision for this dataset + day, in Overview-table vocabulary."""
    if not DECISIONS.exists():
        return "Needs Review"
    log = pd.read_csv(DECISIONS, parse_dates=["date"])
    m = log[(log["dataset"] == dataset_short) & (log["date"] == pd.Timestamp(day).normalize())]
    return STATUS_MAP.get(m.iloc[-1]["decision"], "Needs Review") if not m.empty else "Needs Review"


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
        if r["increase"] > 0 and r["share_of_increase"] >= 0.05:
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
    return f'<span class="pill {cls}"><i class="pill-dot"></i>{label}</span>'


def tag(text: str) -> str:
    return f'<span class="tag">{text}</span>'


def page_head(crumb: str, title: str, sub: str, chip: str) -> None:
    html(f'<div class="page-head"><div><div class="crumb">{crumb}</div><div class="ptitle">{title}</div>'
         f'<div class="psub">{sub}</div></div><div class="chip"><span class="live"></span>{chip}</div></div>')


def kpi(col, icon: str, label: str, value: str, sub: str = "", small: bool = False) -> None:
    html(f'<div class="kpi"><div class="kpi-top"><span class="kpi-icon"><span class="msr">{icon}</span></span>'
         f'<span class="kpi-label">{label}</span></div>'
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


def goto_investigate(day) -> None:
    """Jump straight to Investigate for a given day -- used by every 'Investigate' button
    and clickable table row on Overview, so the DSS acts on a click, not just a read."""
    st.session_state["selected_day"] = day
    st.session_state["page"] = "Investigate"
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

html('<div class="brand"><div class="logo"><span class="msr">query_stats</span></div><div>'
     '<div class="bname">Cost Anomaly DSS</div><div class="bsub">Azure Cost Decision Support</div></div></div>',
     st.sidebar)

for group_label, items in NAV_GROUPS:
    html(f'<div class="navlabel">{group_label}</div>', st.sidebar)
    for name, icon in items:
        if st.sidebar.button(name, icon=icon, key=f"nav_{name.replace(' ', '_')}", use_container_width=True):
            st.session_state["page"] = name
page = st.session_state["page"]
# The active page is known here in Python, so its nav button is highlighted with a
# targeted rule rather than guessed client-side -- no stray highlight is possible.
html(f'<style>section[data-testid="stSidebar"] [class*="st-key-nav_{page.replace(" ", "_")}"] button, '
     f'section[data-testid="stSidebar"] [class*="st-key-nav_{page.replace(" ", "_")}"] button * '
     '{ background:var(--accent-soft) !important; color:var(--accent) !important; font-weight:600 !important; }</style>')

html('<div class="navlabel">Active dataset</div>', st.sidebar)
ds_name = st.sidebar.selectbox("Dataset", list(DATASETS) + ["Upload a CSV"], label_visibility="collapsed")
det, det_label = DET, "Consensus"
html('<div class="sidefoot">Suggests checks only — never changes Azure resources.<br>'
     'NSBM Green University · BSc MIS</div>', st.sidebar)

if ds_name in DATASETS:
    cfg = DATASETS[ds_name]
    W = load(str(cfg["file"]))
else:
    cfg = {"cur": "", "truth": None, "short": "Uploaded data", "key": "upload", "note": "Uploaded data"}
    ups = st.file_uploader("Daily per-service costs — CSV or Excel", type=["csv", "xlsx", "xls"],
                           accept_multiple_files=True)
    if not ups:
        page_head("Data source", "Upload cost files",
                  "Each file needs a date, a service name, and a cost column. Add more than one "
                  "(for example, one export per month) and they're combined automatically.", "Waiting for data")
        st.stop()
    try:
        W = combine_wide([to_wide(read_any(f)) for f in ups])
    except StopIteration:
        st.error("One of the files is missing a date, service name, or cost column.")
        st.stop()
    if len(ups) > 1:
        cfg["short"] = f"Uploaded data ({len(ups)} files)"

cur = cfg["cur"]
def money(v: float) -> str:  # noqa: E302
    return f"{cur}{v:,.2f}"
num_fmt = f"{cur}%.2f"

A = analyse(W)
flagged = A[A[det] == 1]
span = f"{A.index.min():%d %b %Y} – {A.index.max():%d %b %Y}"
chip = f"Monitoring · {det_label}"
platform = st.session_state.get("alert_platform", "Slack")
webhook = st.session_state.get("alert_webhook", "")

# A pending jump from a clicked row/button (Overview or elsewhere) seeds Investigate's
# own selectbox *before* it's created, so Streamlit shows the clicked day immediately
# instead of whatever was last manually picked.
if "selected_day" in st.session_state:
    _want = st.session_state.pop("selected_day")
    if _want in flagged.index:
        st.session_state["investigate_day"] = _want

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    page_head("Overview", "Monitor, detect and understand unusual Azure costs", f"{cfg['short']} · {span}", chip)

    hl, hr1, hr2 = st.columns([5, 2.3, 1], gap="small")
    with hr1:
        date_range = st.date_input("Date range", value=(A.index.min().date(), A.index.max().date()),
                                   min_value=A.index.min().date(), max_value=A.index.max().date(),
                                   label_visibility="collapsed", key="ov_date_range")
    with hr2:
        if st.button("Refresh", icon=":material/refresh:", key="btn_refresh", use_container_width=True):
            load.clear()
            analyse.clear()
            st.rerun()

    flagged_sorted = flagged.sort_index()
    latest_day = flagged_sorted.index[-1] if not flagged_sorted.empty else None
    cost_risk = flagged_sorted.loc[latest_day, "extra"] * 30 if latest_day is not None else 0.0

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "payments", "Total Spend", money(A["total"].sum()), f"{money(A['total'].mean())} per day on average")
    kpi(k2, "warning", "Anomalous Days", f"{len(flagged)}",
        f"{(flagged['pct'] >= 0.75).sum()} high · {((flagged['pct'] >= 0.25) & (flagged['pct'] < 0.75)).sum()} medium severity")
    kpi(k3, "trending_up", "Extra Spend", money(flagged["extra"].sum()),
        f"{flagged['extra'].sum() / A['total'].sum():.1%} of total spend")
    kpi(k4, "shield_moon", "Cost Risk", money(cost_risk) if latest_day is not None else "—",
        "If the latest anomaly continues 30 days" if latest_day is not None else "No anomalies currently")
    st.write("")

    left, right = st.columns([2.15, 1], gap="medium")
    with left.container(key="card_trend"):
        card_head("Daily spend", "Normal = 7-day median",
                  f'<div class="legend"><span><span class="dot" style="background:{ACCENT}"></span>Daily cost</span>'
                  f'<span><span class="dash"></span>Normal</span>'
                  f'<span><span class="dot" style="background:{RED}"></span>Anomaly</span></div>')
        plot = A.reset_index()[["date", "total", "normal", "pct"]].copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            plot = plot[(plot["date"] >= lo) & (plot["date"] <= hi)]
        plot["anomaly"] = (A.loc[plot["date"], det].values == 1) if not plot.empty else []
        plot["diff"] = plot["total"] - plot["normal"]
        plot["status"] = plot["anomaly"].map({True: "Anomaly", False: "Normal"})
        base = alt.Chart(plot).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)))
        # Flat low-opacity fill, not a gradient -- one less visual effect, same read.
        area = base.mark_area(color="rgba(40,118,210,0.08)", line={"color": ACCENT, "strokeWidth": 2},
                              interpolate="monotone").encode(y=alt.Y("total:Q", title=None))
        normal = base.mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(y="normal:Q")
        pts = base.transform_filter("datum.anomaly").mark_circle(
            color=RED, size=64, opacity=1, stroke="white", strokeWidth=2).encode(y="total:Q")
        hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
        rule = base.mark_rule(color="#c5c9d2", strokeWidth=1).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("total:Q", title="Cost", format=",.2f"),
                     alt.Tooltip("normal:Q", title="Normal", format=",.2f"),
                     alt.Tooltip("diff:Q", title="Difference", format="+,.2f"),
                     alt.Tooltip("pct:Q", title="Change", format="+.0%"),
                     alt.Tooltip("status:N", title="Status")]).add_params(hover)
        st.altair_chart(styled(area + normal + pts + rule, 372), use_container_width=True)

    with right:
        with st.container(key="card_latest"):
            card_head("Latest Anomaly")
            if latest_day is None:
                html('<div class="csub">No anomalies detected in this period.</div>')
            else:
                lr = flagged_sorted.loc[latest_day]
                lsvc = top_driver(W, latest_day)
                html(f'<div class="b-title" style="margin-top:0">{latest_day:%d %b %Y}</div>'
                     f'<div class="csub" style="margin:2px 0 10px 0">{lsvc}</div>'
                     f'<div class="stats" style="gap:18px">'
                     f'<div><div class="stat-v" style="color:var(--red)">+{money(lr["extra"])}</div></div>'
                     f'<div><div class="stat-v">{lr["pct"]:+.0%}</div></div>'
                     f'<div>{pill(lr["pct"])}</div></div>'
                     '<div class="csub" style="margin-top:10px">Unusual increase compared with normal spending.</div>')
                if st.button("Investigate", icon=":material/troubleshoot:", key="btn_inv_latest",
                            type="primary", use_container_width=True):
                    goto_investigate(latest_day)
        st.write("")
        with st.container(key="card_action"):
            card_head("Recommended Action")
            if latest_day is None:
                html('<div class="csub">Nothing to review right now.</div>')
            else:
                drv = explain_day(W, latest_day, 3)
                top_a = drv[drv["increase"] > 0].head(2)
                html(f'<div class="ctitle" style="font-size:var(--text-sm)">Review {lsvc} costs</div>'
                     f'<div class="csub" style="margin:4px 0 10px 0">An unusual increase of '
                     f'{money(lr["extra"])} was detected.</div>')
                html("".join(f'<div class="step"><div class="num">{i}</div>'
                             f'<div><span>{ACTIONS.get(s, DEFAULT_ACTION)}</span></div></div>'
                             for i, s in enumerate(top_a.index, 1)))
                if st.button("Investigate", icon=":material/troubleshoot:", key="btn_inv_action",
                            use_container_width=True):
                    goto_investigate(latest_day)

    st.write("")
    with st.container(key="card_recent"):
        card_head("Recent Anomalies", f"Latest {min(10, len(flagged))} of {len(flagged)} — click a row to investigate")
        if flagged.empty:
            html('<div class="csub">No anomalous days with the current method.</div>')
        else:
            recent = flagged.iloc[::-1].head(10)
            dates = list(recent.index)
            tbl = pd.DataFrame({
                "Date": [d.date() for d in dates],
                "Service": [top_driver(W, d) for d in dates],
                "Extra Cost": recent["extra"].values,
                "Severity": [severity(p)[0] for p in recent["pct"]],
                "Status": [decision_status(cfg["short"], d) for d in dates],
            })
            sel = st.dataframe(tbl, hide_index=True, width="stretch", on_select="rerun",
                               selection_mode="single-row", key="recent_tbl", column_config={
                                   "Date": st.column_config.DateColumn(format="D MMM YYYY"),
                                   "Extra Cost": st.column_config.NumberColumn(format=num_fmt),
                               })
            rows = sel["selection"]["rows"] if isinstance(sel, dict) else sel.selection.rows
            if rows:
                goto_investigate(dates[rows[0]])
            with st.expander("Full detail"):
                full = pd.DataFrame({
                    "Date": flagged.index.date,
                    "Severity": [severity(p)[0] for p in flagged["pct"]],
                    "Cost": flagged["total"].values, "Normal": flagged["normal"].values,
                    "Extra cost": flagged["extra"].values, "Increase": (flagged["pct"] * 100).values,
                    "Main driver": [top_driver(W, d) for d in flagged.index],
                    "Detectors agreeing": flagged["votes"].values,
                }).iloc[::-1]
                st.dataframe(full, hide_index=True, width="stretch", column_config={
                    "Date": st.column_config.DateColumn(format="D MMM YYYY"),
                    "Cost": st.column_config.NumberColumn(format=num_fmt),
                    "Normal": st.column_config.NumberColumn(format=num_fmt),
                    "Extra cost": st.column_config.NumberColumn(format=num_fmt),
                    "Increase": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                max_value=float(max(100, full["Increase"].max()))),
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
                                               f"{severity(A.loc[d, 'pct'])[0]}", key="investigate_day")
    r = A.loc[day]
    drivers = explain_day(W, day, 8)
    top = drivers[drivers["increase"] > 0].head(3)

    html(f'<div class="crumb">Anomaly Summary</div>')
    html(f'<div class="banner"><div>{pill(r["pct"])}<div class="b-title">Cost anomaly on {day:%A, %d %B %Y}</div>'
         f'<div class="csub">Flagged by {int(r["votes"])} of 4 detectors</div></div>'
         f'<div class="stats"><div><div class="stat-l">Actual Cost</div><div class="stat-v">{money(r["total"])}</div></div>'
         f'<div><div class="stat-l">Normal Cost</div><div class="stat-v">{money(r["normal"])}</div></div>'
         f'<div><div class="stat-l">Extra Cost</div><div class="stat-v">+{money(r["extra"])}</div></div>'
         f'<div><div class="stat-l">Change %</div><div class="stat-v">{r["pct"]:+.0%}</div></div>'
         f'<div><div class="stat-l">Financial Impact (30d)</div><div class="stat-v">{money(r["extra"] * 30)}</div></div>'
         f'</div></div>')

    left, right = st.columns([1.618, 1], gap="medium")  # golden ratio: drivers chart carries the primary evidence
    with right.container(key="card_why"):
        card_head("Why was this flagged?")
        if not top.empty:
            lead = top.iloc[0]
            new = " New service — not in use before." if lead["normal"] < 1e-6 else ""
            html(f'<div class="callout"><span class="msr">lightbulb</span><div><b>{top.index[0]}</b> drives '
                 f'<b>{lead["share_of_increase"]:.0%}</b> of the increase: +{money(lead["increase"])} over its '
                 f'normal {money(lead["normal"])}.{new}</div></div>')
        card_head("Recommended Checks")
        html("".join(f'<div class="step"><div class="num">{i}</div><div><b>{svc}</b><br>'
                     f'<span>{ACTIONS.get(svc, DEFAULT_ACTION)}</span></div></div>'
                     for i, svc in enumerate(top.index, 1)))

    with left.container(key="card_drivers"):
        card_head("Cost Driver Ranking", "Change vs normal, per service")
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
        with st.expander("Explanation: SHAP model view"):
            sv = shap_for(W, day).sort_values().head(8).rename_axis("service").rename("shap").reset_index()
            html('<div class="csub">An ML-based ranking. The chart above was more accurate in testing.</div>')
            st.altair_chart(styled(alt.Chart(sv).mark_bar(color=ACCENT, cornerRadiusEnd=4, height=14).encode(
                x=alt.X("shap:Q", title=None), y=alt.Y("service:N", sort="x", title=None),
                tooltip=[alt.Tooltip("service", title="Service"), alt.Tooltip("shap:Q", format=".3f", title="SHAP")]),
                30 * len(sv)).configure_axisY(grid=False), use_container_width=True)

    st.write("")
    with st.container(key="card_decision"):
        card_head("Record Decision", "Record how this was handled")
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
                if ok:
                    st.success("Alert sent.")
                else:
                    st.error(f"Alert failed: {info}")
            else:
                st.warning("Add a webhook URL on the Alerts page.")
        if DECISIONS.exists():
            log = pd.read_csv(DECISIONS)
            html(f'<div class="csub" style="margin-top:14px">Decision log · {len(log)} entries</div>')
            st.dataframe(log.iloc[::-1].head(10), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Model Evaluation
# --------------------------------------------------------------------------- #
elif page == "Model Evaluation":
    page_head("Model Evaluation", "How well does it work?",
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
        kpi(k1, "radar", "Incidents detected", f"{(res['Detected'] == 'Yes').sum()} of {len(res)}", det_label)
        kpi(k2, "target", "Driver named correctly", f"{(res['Driver correct'] == 'Yes').sum()} of {len(res)}",
            "First-ranked service")
        if dsum is not None:
            both = dsum.groupby("detector")["f1"].mean()
            kpi(k3, "insights", "Detection model in use", "Consensus (2 of 4)",
                f"F1 {both['consensus']:.2f} averaged over both datasets, best of {len(both)}", small=True)
        if asum is not None:
            pooled = asum[asum.dataset == "kaggle"].groupby("method")["top1"].mean()
            kpi(k4, "psychology", "Best explanation method", METHOD_NAMES[pooled.idxmax()],
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
            for ds, label in [("startup", "Case startup"), ("kaggle", "Public Azure (real daily)")]:
                col = st
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
    mlx_file = EVAL / "ml_evaluation.csv"
    if dsum is not None and mlx_file.exists():
        det_runs = pd.read_csv(EVAL / "detection_runs.csv")
        mlx = pd.read_csv(mlx_file)
        ml = ["isolation_forest", "one_class_svm", "lof", "kmeans"]
        with st.container(key="card_ml"):
            card_head("Machine-learning model evaluation",
                      "Four unsupervised ML models, same features, threshold-free and deployment-style tests")
            for ds, label in [("startup", "Case startup"), ("kaggle", "Public Azure (real daily)")]:
                col = st
                r = det_runs[(det_runs.dataset == ds) & det_runs.detector.isin(ml + ["consensus"])]
                g = r.groupby("detector").agg(roc=("roc_auc", "mean"), pr=("pr_auc", "mean"))
                g["f1"] = r[r.k == 0.3].groupby("detector")["f1"].mean()
                wf = mlx[(mlx.dataset == ds) & (mlx.experiment == "walk_forward")].groupby("detector")["f1"].mean()
                g["walk"] = wf.reindex(g.index).map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
                g = g.sort_values("roc", ascending=False)
                col.markdown(f"**{label}**")
                col.dataframe(pd.DataFrame({
                    "Model": g.index.map(DETECTOR_NAMES), "ROC-AUC": g["roc"].values, "PR-AUC": g["pr"].values,
                    "F1": g["f1"].values, "F1 walk-forward": g["walk"].values}),
                    hide_index=True, width="stretch", column_config={
                        "ROC-AUC": st.column_config.ProgressColumn(format="%.2f", min_value=0.5, max_value=1.0,
                                                                   help="0.5 = random ranking, 1.0 = perfect"),
                        "PR-AUC": st.column_config.NumberColumn(format="%.2f"),
                        "F1": st.column_config.NumberColumn(format="%.2f", help="Incidents of +30%, fitted on the whole period"),
                        "F1 walk-forward": st.column_config.TextColumn(
                            help="Incidents of +30%, retrained weekly on past days only (deployment-style)")})
            html('<div class="csub" style="margin-top:8px">In the main evaluation Isolation Forest is the strongest '
                 'ML model on both datasets (Wilcoxon p &lt; 0.01 against each other ML model), so it is the ML '
                 'member of the consensus detector. Retrained weekly on past days only, all four lose accuracy on '
                 'the real public data; on the case startup data LOF and K-Means improve.</div>')
        st.write("")
    with st.expander("Figures"):
        for img in ("fig_detection_ml.png", "fig_detection_stat.png", "fig_auc.png", "fig_attribution.png",
                   "fig_financial_impact.png"):
            if (EVAL / img).exists():
                st.image(str(EVAL / img), width="stretch")
    if dsum is None:
        st.warning("Run `python scripts/evaluate.py` to generate the evaluation results.")

# --------------------------------------------------------------------------- #
# Real Cases
# --------------------------------------------------------------------------- #
elif page == "Real Cases":
    page_head("Real Cases", "Real cost spikes, explained",
              "Actual Azure billing — no injected data", chip)

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
                      tag("Real case") + " " + pill(after / before - 1).replace("</span>", f" · {after / before - 1:+.0%}</span>"))
            html(bars_html(c["change"][c["change"] > 0.005]))
            html('<div class="csub" style="margin-top:10px">A new AI agent (search grounding and GPT tokens) '
                 'drove the spike.</div>')
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
                      tag("Real case") + " " + pill(rr["pct"]).replace("</span>", f" · {rr['pct']:+.0%}</span>"))
            html(bars_html(ex["increase"].clip(lower=0)))
        if i % 2 == 1:
            st.write("")

# --------------------------------------------------------------------------- #
# Data Source
# --------------------------------------------------------------------------- #
elif page == "Data Source":
    page_head("Data Source", "Manage and validate your Azure billing data", f"{cfg['short']} · {span}", chip)

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "calendar_month", "Days Covered", f"{len(A)}", span)
    kpi(k2, "dns", "Services Tracked", f"{W.shape[1]}", "Distinct Azure services")
    kpi(k3, "table_rows", "Records", f"{len(A) * W.shape[1]:,}", "Date × service rows")
    kpi(k4, "verified", "Validation", "Passed", "Date, service and cost columns found")
    st.write("")
    with st.container(key="card_datasource"):
        card_head("Source details")
        rows = [("Dataset", cfg["short"]), ("Date range", span),
               ("Currency", cur if cur else "Billing currency (as exported)"),
               ("Services", str(W.shape[1])),
               ("Source type", "Uploaded file(s)" if cfg["key"] == "upload" else "Built-in sample dataset"),
               ("Upload status", "Not applicable" if cfg["key"] != "upload" else f"{len(ups)} file(s) loaded"),
               ("Validation status", "Passed — date, service and cost columns found in every file")]
        html("".join(f'<div class="item"><div class="i-date">{k}</div><div class="i-amt">{v}</div></div>'
                    for k, v in rows))
    st.write("")
    with st.container(key="card_switch"):
        card_head("Switch dataset", "The active dataset can also be changed from the sidebar at any time")
        html('<div class="csub">Case startup — real billing, with known incidents. Public Azure — real daily '
             'billing (Kaggle, 2023). Or upload your own CSV/Excel export(s).</div>')

# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
elif page == "Alerts":
    page_head("Alerts", "Notify your team", "Send a Slack or Microsoft Teams message for a flagged day", chip)
    with st.container(key="card_alerts"):
        card_head("Alert channel", "Used by the \"Send alert\" button on the Investigate page")
        c1, c2 = st.columns(2)
        # Streamlit clears a widget's session_state entry once its page stops being
        # rendered, so the value is copied into a plain (non-widget-keyed) session_state
        # entry below -- that one survives switching to other pages, the widget's own
        # key would not.
        plats = ["Slack", "Microsoft Teams"]
        pf = c1.selectbox("Platform", plats, index=plats.index(st.session_state.get("alert_platform", "Slack")),
                          key="alert_platform_widget")
        wh = c2.text_input("Webhook URL", type="password", placeholder="https://hooks...",
                           value=st.session_state.get("alert_webhook", ""), key="alert_webhook_widget")
        st.session_state["alert_platform"], st.session_state["alert_webhook"] = pf, wh
        html('<div class="csub" style="margin-top:8px">Create an incoming webhook in Slack or Microsoft Teams, '
             'paste its URL here, then use "Send alert" on the Investigate page for any flagged day. The message '
             'includes the date, severity, extra cost, and top cost drivers.</div>')

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
else:
    page_head("Settings", "How the system behaves", "Read-only — fixed by the evaluation, not user-configurable", chip)
    with st.container(key="card_detection_settings"):
        card_head("Detection model")
        html('<div class="modelbox"><b>Consensus detector</b><br>A day is flagged when at least 2 of 4 detectors '
             'agree: Isolation Forest, median + MAD, moving-average band and 3-sigma. Chosen for the best F1 '
             'across both evaluation datasets — see Model Evaluation for the comparison.</div>')
    st.write("")
    with st.container(key="card_privacy_settings"):
        card_head("Data handling")
        html('<div class="csub">Uploaded files are processed in memory for this session only and are not stored. '
             'Decisions you record on the Investigate page are saved locally to results/decision_log.csv.</div>')
    st.write("")
    with st.container(key="card_about_settings"):
        card_head("About")
        html('<div class="csub">Azure Cost Anomaly DSS — suggests checks only, never changes Azure resources.<br>'
             'BSc MIS final-year research prototype, NSBM Green University.</div>')
