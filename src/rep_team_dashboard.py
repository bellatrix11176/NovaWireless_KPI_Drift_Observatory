"""
NovaWireless KPI Drift Observatory — Rep Team Performance Dashboard
Reads: data/external/calls_sanitized_2025-*.csv
Teams: 250 reps split into 10 teams of 25 (sorted by rep_id)
Run: streamlit run src/rep_team_dashboard.py
"""

import os
import glob
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # one level up from src/
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "external")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TEAM_SIZE = 25
BOOL_COLS = [
    "is_repeat_call", "imei_mismatch_flag", "nrf_generated_flag",
    "promo_override_post_call", "line_added_no_usage_flag",
    "line_added_same_day_store", "rep_aware_gaming", "true_resolution",
    "resolution_flag", "repeat_contact_30d", "repeat_contact_31_60d",
    "escalation_flag", "credit_applied", "credit_authorized",
    "customer_is_churned",
]

FRAUD_SCENARIOS = {
    "fraud_store_promo", "fraud_line_add",
    "fraud_hic_exchange", "fraud_care_promo",
}

PALETTE = {
    "primary":   "#00C2CB",
    "danger":    "#FF4C61",
    "warning":   "#FFB347",
    "success":   "#4ECDC4",
    "neutral":   "#8892A4",
    "bg":        "#0D1117",
    "surface":   "#161B22",
    "surface2":  "#21262D",
    "text":      "#E6EDF3",
    "subtext":   "#8892A4",
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading sanitized call files…")
def load_all_files(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "calls_sanitized_2025-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        st.error(f"No files found at: {pattern}")
        st.stop()
    frames = []
    for f in files:
        tmp = pd.read_csv(f, low_memory=False)
        tmp["source_file"] = os.path.basename(f)
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)

    # Parse dates
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    df["month"] = df["call_date"].dt.to_period("M").astype(str)
    df["month_num"] = df["call_date"].dt.month

    # Coerce boolean columns that might be strings
    for col in BOOL_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            df[col] = df[col].astype("boolean")

    # Numeric safety
    for col in ["aht_secs", "rep_gaming_propensity", "rep_burnout_level",
                "rep_policy_skill", "credit_amount", "customer_trust_baseline",
                "customer_churn_risk_effective", "agent_qa_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def assign_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Sort reps by rep_id, assign teams of TEAM_SIZE sequentially."""
    sorted_reps = sorted(df["rep_id"].unique())
    team_map = {
        rep_id: f"Team {(i // TEAM_SIZE) + 1:02d}"
        for i, rep_id in enumerate(sorted_reps)
    }
    df["team"] = df["rep_id"].map(team_map)
    return df, team_map


# ── METRIC HELPERS ────────────────────────────────────────────────────────────
def bool_mean(series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if len(s.dropna()) > 0 else 0.0


def compute_rep_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["rep_id", "rep_name", "team"])

    metrics = g.agg(
        total_calls        = ("call_id",                   "count"),
        avg_aht_secs       = ("aht_secs",                  "mean"),
        resolution_proxy   = ("resolution_flag",           lambda x: bool_mean(x)),
        resolution_true    = ("true_resolution",           lambda x: bool_mean(x)),
        repeat_30d         = ("repeat_contact_30d",        lambda x: bool_mean(x)),
        repeat_31_60d      = ("repeat_contact_31_60d",     lambda x: bool_mean(x)),
        escalation_rate    = ("escalation_flag",           lambda x: bool_mean(x)),
        credit_rate        = ("credit_applied",            lambda x: bool_mean(x)),
        avg_gaming         = ("rep_gaming_propensity",     "mean"),
        avg_burnout        = ("rep_burnout_level",         "mean"),
        avg_policy_skill   = ("rep_policy_skill",          "mean"),
        avg_qa_score       = ("agent_qa_score",            "mean"),
        avg_trust_baseline = ("customer_trust_baseline",   "mean"),
        avg_churn_risk     = ("customer_churn_risk_effective", "mean"),
    ).reset_index()

    # Derived
    metrics["resolution_gap"]  = metrics["resolution_proxy"] - metrics["resolution_true"]
    metrics["avg_aht_min"]     = metrics["avg_aht_secs"] / 60
    metrics["bandaid_count"]   = df[df["credit_type"] == "bandaid"].groupby("rep_id")["call_id"].count().reindex(metrics["rep_id"]).fillna(0).values
    metrics["bandaid_fail"]    = (
        df[(df["credit_type"] == "bandaid") & (df["repeat_contact_31_60d"] == True)]
        .groupby("rep_id")["call_id"].count()
        .reindex(metrics["rep_id"]).fillna(0).values
    )
    metrics["bandaid_fail_rate"] = np.where(
        metrics["bandaid_count"] > 0,
        metrics["bandaid_fail"] / metrics["bandaid_count"],
        0.0
    )
    fraud_counts = (
        df[df["scenario"].isin(FRAUD_SCENARIOS)]
        .groupby("rep_id")["call_id"].count()
        .reindex(metrics["rep_id"]).fillna(0)
    )
    metrics["fraud_involvement"] = fraud_counts.values / metrics["total_calls"].clip(lower=1)

    return metrics


def compute_team_metrics(rep_metrics: pd.DataFrame) -> pd.DataFrame:
    tm = rep_metrics.groupby("team").agg(
        reps                = ("rep_id",             "count"),
        total_calls         = ("total_calls",         "sum"),
        avg_aht_min         = ("avg_aht_min",         "mean"),
        resolution_proxy    = ("resolution_proxy",    "mean"),
        resolution_true     = ("resolution_true",     "mean"),
        resolution_gap      = ("resolution_gap",      "mean"),
        repeat_30d          = ("repeat_30d",          "mean"),
        repeat_31_60d       = ("repeat_31_60d",       "mean"),
        escalation_rate     = ("escalation_rate",     "mean"),
        avg_gaming          = ("avg_gaming",          "mean"),
        avg_burnout         = ("avg_burnout",         "mean"),
        avg_policy_skill    = ("avg_policy_skill",    "mean"),
        bandaid_fail_rate   = ("bandaid_fail_rate",   "mean"),
        fraud_involvement   = ("fraud_involvement",   "mean"),
        avg_trust_baseline  = ("avg_trust_baseline",  "mean"),
        avg_churn_risk      = ("avg_churn_risk",      "mean"),
    ).reset_index()
    return tm


def compute_monthly_rep(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["rep_id", "rep_name", "team", "month", "month_num"])
    return g.agg(
        calls              = ("call_id",              "count"),
        avg_aht_secs       = ("aht_secs",             "mean"),
        resolution_proxy   = ("resolution_flag",      lambda x: bool_mean(x)),
        resolution_true    = ("true_resolution",      lambda x: bool_mean(x)),
        avg_gaming         = ("rep_gaming_propensity","mean"),
        avg_burnout        = ("rep_burnout_level",    "mean"),
        repeat_31_60d      = ("repeat_contact_31_60d",lambda x: bool_mean(x)),
    ).reset_index().sort_values("month_num")


# ── STREAMLIT LAYOUT ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaWireless · Rep Performance",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
      background-color: {PALETTE['bg']};
      color: {PALETTE['text']};
  }}
  h1, h2, h3 {{
      font-family: 'Space Mono', monospace;
      letter-spacing: -0.03em;
  }}
  .metric-card {{
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['surface2']};
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 8px;
  }}
  .metric-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: {PALETTE['subtext']};
      font-family: 'Space Mono', monospace;
  }}
  .metric-value {{
      font-size: 28px;
      font-weight: 600;
      line-height: 1.1;
      margin-top: 4px;
  }}
  .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
  }}
  .stSelectbox label, .stMultiSelect label, .stSlider label {{
      color: {PALETTE['subtext']} !important;
      font-size: 12px;
      font-family: 'Space Mono', monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
  div[data-testid="stSidebar"] {{ background-color: {PALETTE['surface']}; }}
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_raw = load_all_files(DATA_DIR)
df_raw, team_map = assign_teams(df_raw)

rep_metrics  = compute_rep_metrics(df_raw)
team_metrics = compute_team_metrics(rep_metrics)
monthly_rep  = compute_monthly_rep(df_raw)

all_teams  = sorted(rep_metrics["team"].unique())
all_months = sorted(df_raw["month"].unique())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 NovaWireless\n### KPI Drift Observatory")
    st.markdown("---")

    selected_team = st.selectbox("Select Team", ["All Teams"] + all_teams)

    month_filter = st.multiselect(
        "Month Filter",
        options=all_months,
        default=all_months,
    )

    st.markdown("---")
    st.markdown("**View**")
    view_mode = st.radio(
        "Dashboard View",
        ["Team Overview", "Rep Drilldown", "Drift Over Time", "Fraud & Bandaid", "Governance Signals"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(f"📁 Source: `data/external/`  \n{len(all_months)} months · {len(df_raw):,} rows · 250 reps · 10 teams")


# ── FILTER DATA ───────────────────────────────────────────────────────────────
df_f = df_raw[df_raw["month"].isin(month_filter)].copy()

if selected_team != "All Teams":
    rep_metrics_f  = rep_metrics[rep_metrics["team"] == selected_team]
    team_metrics_f = team_metrics[team_metrics["team"] == selected_team]
    monthly_rep_f  = monthly_rep[monthly_rep["team"] == selected_team]
    df_f           = df_f[df_f["team"] == selected_team]
else:
    rep_metrics_f  = rep_metrics.copy()
    team_metrics_f = team_metrics.copy()
    monthly_rep_f  = monthly_rep.copy()


# helper: recompute rep metrics on filtered data
rep_filtered = compute_rep_metrics(df_f)
if selected_team != "All Teams":
    rep_filtered = rep_filtered[rep_filtered["team"] == selected_team]


def pct(val): return f"{val*100:.1f}%"
def mins(val): return f"{val:.1f} min"


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: TEAM OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Team Overview":
    st.markdown("# Team Overview")
    st.markdown(f"*{selected_team} · {', '.join(month_filter) if len(month_filter) <= 3 else f'{len(month_filter)} months selected'}*")

    # KPI strip
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    total_calls = len(df_f)
    avg_res_proxy = bool_mean(df_f["resolution_flag"])
    avg_res_true  = bool_mean(df_f["true_resolution"])
    gap           = avg_res_proxy - avg_res_true
    avg_gaming    = df_f["rep_gaming_propensity"].mean()
    bandaid_pct   = (df_f["credit_type"] == "bandaid").sum() / max(len(df_f), 1)

    with col1:
        st.metric("Total Calls", f"{total_calls:,}")
    with col2:
        st.metric("Proxy Resolution", pct(avg_res_proxy))
    with col3:
        st.metric("True Resolution", pct(avg_res_true))
    with col4:
        st.metric("Resolution Gap", pct(gap), delta=f"{gap*100:+.1f}pp", delta_color="inverse")
    with col5:
        st.metric("Avg Gaming Propensity", f"{avg_gaming:.3f}")
    with col6:
        st.metric("Bandaid Rate", pct(bandaid_pct))

    st.markdown("---")

    # Team comparison table
    st.subheader("Team Comparison Matrix")
    display_cols = ["team", "total_calls", "avg_aht_min", "resolution_proxy",
                    "resolution_true", "resolution_gap", "avg_gaming",
                    "avg_burnout", "bandaid_fail_rate", "fraud_involvement"]
    tc = team_metrics[display_cols].copy()
    for c in ["resolution_proxy","resolution_true","resolution_gap",
              "avg_gaming","avg_burnout","bandaid_fail_rate","fraud_involvement"]:
        tc[c] = tc[c].map(lambda x: f"{x:.3f}")
    tc["avg_aht_min"] = tc["avg_aht_min"].map(lambda x: f"{x:.1f}")
    st.dataframe(tc.rename(columns={
        "team":"Team","total_calls":"Calls","avg_aht_min":"AHT (min)",
        "resolution_proxy":"Proxy Res","resolution_true":"True Res",
        "resolution_gap":"Gap","avg_gaming":"Gaming","avg_burnout":"Burnout",
        "bandaid_fail_rate":"Bandaid Fail","fraud_involvement":"Fraud Inv."
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Resolution Gap by Team")
        fig = px.bar(
            team_metrics.sort_values("resolution_gap", ascending=False),
            x="team", y="resolution_gap",
            color="resolution_gap",
            color_continuous_scale=["#4ECDC4", "#FFB347", "#FF4C61"],
            labels={"resolution_gap": "Gap (proxy − true)", "team": "Team"},
        )
        fig.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"], showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Gaming Propensity vs Burnout by Team")
        fig2 = px.scatter(
            team_metrics,
            x="avg_gaming", y="avg_burnout",
            size="total_calls", color="resolution_gap",
            text="team",
            color_continuous_scale=["#4ECDC4", "#FF4C61"],
            labels={"avg_gaming": "Avg Gaming Propensity",
                    "avg_burnout": "Avg Burnout",
                    "resolution_gap": "Resolution Gap"},
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scenario distribution across teams
    st.subheader("Scenario Distribution by Team")
    scen_team = df_raw.groupby(["team", "scenario"])["call_id"].count().reset_index()
    scen_team.columns = ["team", "scenario", "count"]
    pivot = scen_team.pivot(index="team", columns="scenario", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)
    fig3 = px.bar(
        pivot_pct.reset_index().melt(id_vars="team"),
        x="team", y="value", color="scenario",
        labels={"value": "Share of Calls", "team": "Team", "scenario": "Scenario"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig3.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: REP DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Rep Drilldown":
    st.markdown("# Rep Drilldown")

    team_for_rep = st.selectbox(
        "Team",
        options=all_teams,
        index=0 if selected_team == "All Teams" else all_teams.index(selected_team),
        key="rep_team_sel"
    )

    reps_in_team = rep_filtered[rep_filtered["team"] == team_for_rep].sort_values("rep_id")
    rep_options  = reps_in_team["rep_name"].tolist()
    selected_rep = st.selectbox("Select Rep", rep_options, key=f"rep_sel_{team_for_rep}")

    # Guard: if selected_rep somehow not in this team, fall back to first
    match = reps_in_team[reps_in_team["rep_name"] == selected_rep]
    if len(match) == 0:
        match = reps_in_team.iloc[[0]]
    rep_row = match.iloc[0]
    rep_id  = rep_row["rep_id"]

    # Rep KPI strip
    st.markdown("---")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Calls Handled",    f"{int(rep_row['total_calls']):,}")
    c2.metric("Proxy Resolution", pct(rep_row["resolution_proxy"]))
    c3.metric("True Resolution",  pct(rep_row["resolution_true"]))
    c4.metric("Resolution Gap",   pct(rep_row["resolution_gap"]),
              delta=f"{rep_row['resolution_gap']*100:+.1f}pp", delta_color="inverse")
    c5.metric("Avg Gaming",       f"{rep_row['avg_gaming']:.3f}")
    c6.metric("Avg Burnout",      f"{rep_row['avg_burnout']:.3f}")

    c7,c8,c9,c10,c11,c12 = st.columns(6)
    c7.metric("AHT (min)",         mins(rep_row["avg_aht_min"]))
    c8.metric("Repeat 30d",        pct(rep_row["repeat_30d"]))
    c9.metric("Repeat 31–60d",     pct(rep_row["repeat_31_60d"]))
    c10.metric("Bandaid Fail Rate",pct(rep_row["bandaid_fail_rate"]))
    c11.metric("Fraud Involvement",pct(rep_row["fraud_involvement"]))
    c12.metric("Policy Skill",     f"{rep_row['avg_policy_skill']:.3f}")

    st.markdown("---")

    # Monthly trend for this rep
    rep_monthly = monthly_rep[monthly_rep["rep_id"] == rep_id].sort_values("month_num")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Resolution: Proxy vs True (Monthly)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rep_monthly["month"], y=rep_monthly["resolution_proxy"],
            name="Proxy", line=dict(color=PALETTE["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=rep_monthly["month"], y=rep_monthly["resolution_true"],
            name="True", line=dict(color=PALETTE["danger"], width=2, dash="dash"),
        ))
        fig.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"], yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Gaming Propensity & Burnout Drift")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rep_monthly["month"], y=rep_monthly["avg_gaming"],
            name="Gaming", line=dict(color=PALETTE["warning"], width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=rep_monthly["month"], y=rep_monthly["avg_burnout"],
            name="Burnout", line=dict(color=PALETTE["danger"], width=2, dash="dot"),
        ))
        fig2.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"],
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scenario breakdown for this rep
    rep_calls = df_f[df_f["rep_id"] == rep_id]
    st.subheader("Call Volume by Scenario")
    scen_counts = rep_calls["scenario"].value_counts().reset_index()
    scen_counts.columns = ["scenario", "count"]
    colors = [PALETTE["danger"] if s in FRAUD_SCENARIOS else PALETTE["primary"]
              for s in scen_counts["scenario"]]
    fig3 = go.Figure(go.Bar(
        x=scen_counts["scenario"], y=scen_counts["count"],
        marker_color=colors,
    ))
    fig3.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], xaxis_tickangle=-30,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Compare rep vs team average
    st.subheader(f"Rep vs {team_for_rep} Average")
    team_avg = rep_filtered[rep_filtered["team"] == team_for_rep][
        ["resolution_proxy","resolution_true","avg_gaming","avg_burnout",
         "repeat_31_60d","bandaid_fail_rate","escalation_rate"]
    ].mean()

    compare_metrics = ["resolution_proxy","resolution_true","avg_gaming",
                       "avg_burnout","repeat_31_60d","bandaid_fail_rate","escalation_rate"]
    rep_vals  = [rep_row[m] for m in compare_metrics]
    team_vals = [team_avg[m] for m in compare_metrics]
    labels    = ["Proxy Res","True Res","Gaming","Burnout",
                 "Repeat 31-60d","Bandaid Fail","Escalation"]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(
        r=rep_vals, theta=labels, fill="toself",
        name=selected_rep, line_color=PALETTE["primary"],
    ))
    fig4.add_trace(go.Scatterpolar(
        r=team_vals, theta=labels, fill="toself",
        name=f"{team_for_rep} Avg", line_color=PALETTE["neutral"],
        opacity=0.5,
    ))
    fig4.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor=PALETTE["surface2"],
        ),
        paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"],
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Full team roster table
    st.subheader(f"{team_for_rep} — All 25 Reps")
    display = reps_in_team[[
        "rep_name","total_calls","avg_aht_min","resolution_proxy",
        "resolution_true","resolution_gap","avg_gaming","avg_burnout",
        "repeat_31_60d","bandaid_fail_rate","fraud_involvement"
    ]].copy()
    for c in ["resolution_proxy","resolution_true","resolution_gap",
              "avg_gaming","avg_burnout","repeat_31_60d",
              "bandaid_fail_rate","fraud_involvement"]:
        display[c] = display[c].map(lambda x: f"{x:.3f}")
    display["avg_aht_min"] = display["avg_aht_min"].map(lambda x: f"{x:.1f}")
    st.dataframe(display.rename(columns={
        "rep_name":"Rep","total_calls":"Calls","avg_aht_min":"AHT(min)",
        "resolution_proxy":"Proxy","resolution_true":"True","resolution_gap":"Gap",
        "avg_gaming":"Gaming","avg_burnout":"Burnout","repeat_31_60d":"Repeat 31-60d",
        "bandaid_fail_rate":"Bandaid Fail","fraud_involvement":"Fraud Inv."
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: DRIFT OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Drift Over Time":
    st.markdown("# Drift Over Time")

    team_for_drift = st.selectbox(
        "Focus Team",
        ["All Teams"] + all_teams,
        index=0 if selected_team == "All Teams" else all_teams.index(selected_team) + 1,
    )

    if team_for_drift != "All Teams":
        mdf = monthly_rep[monthly_rep["team"] == team_for_drift]
    else:
        mdf = monthly_rep.copy()

    # Monthly aggregates
    monthly_agg = mdf.groupby(["month","month_num"]).agg(
        resolution_proxy = ("resolution_proxy","mean"),
        resolution_true  = ("resolution_true","mean"),
        avg_gaming       = ("avg_gaming","mean"),
        avg_burnout      = ("avg_burnout","mean"),
        repeat_31_60d    = ("repeat_31_60d","mean"),
    ).reset_index().sort_values("month_num")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resolution Rate Drift")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_agg["month"], y=monthly_agg["resolution_proxy"],
            name="Proxy", mode="lines+markers",
            line=dict(color=PALETTE["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=monthly_agg["month"], y=monthly_agg["resolution_true"],
            name="True", mode="lines+markers",
            line=dict(color=PALETTE["danger"], width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=monthly_agg["month"],
            y=monthly_agg["resolution_proxy"] - monthly_agg["resolution_true"],
            name="Gap", fill="tozeroy", mode="lines",
            line=dict(color=PALETTE["warning"], width=1),
            fillcolor="rgba(255,179,71,0.15)",
        ))
        fig.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"], yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Gaming Propensity Drift")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=monthly_agg["month"], y=monthly_agg["avg_gaming"],
            name="Gaming", mode="lines+markers",
            line=dict(color=PALETTE["warning"], width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=monthly_agg["month"], y=monthly_agg["avg_burnout"],
            name="Burnout", mode="lines+markers",
            line=dict(color=PALETTE["danger"], width=2, dash="dot"),
        ))
        fig2.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"],
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: rep gaming over time (selected team)
    heat_team = team_for_drift if team_for_drift != "All Teams" else all_teams[0]
    st.subheader(f"Gaming Propensity Heatmap — {heat_team}")
    hm_data = monthly_rep[monthly_rep["team"] == heat_team].pivot(
        index="rep_name", columns="month", values="avg_gaming"
    ).fillna(0)
    fig3 = px.imshow(
        hm_data,
        color_continuous_scale=["#161B22", "#FFB347", "#FF4C61"],
        labels=dict(color="Gaming"),
        aspect="auto",
    )
    fig3.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Repeat contact drift
    st.subheader("Repeat Contact 31–60d Rate (Monthly)")
    fig4 = px.line(
        monthly_agg, x="month", y="repeat_31_60d",
        markers=True,
        labels={"repeat_31_60d": "Repeat Contact Rate (31–60d)", "month": "Month"},
        color_discrete_sequence=[PALETTE["danger"]],
    )
    fig4.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: FRAUD & BANDAID
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Fraud & Bandaid":
    st.markdown("# Fraud & Bandaid Analysis")

    fraud_df = df_f[df_f["scenario"].isin(FRAUD_SCENARIOS)]
    bandaid_df = df_f[df_f["credit_type"] == "bandaid"]
    bandaid_fail_df = bandaid_df[bandaid_df["repeat_contact_31_60d"] == True]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud-Scenario Calls", f"{len(fraud_df):,}")
    col2.metric("Fraud Rate",  pct(len(fraud_df) / max(len(df_f), 1)))
    col3.metric("Bandaid Credits", f"{len(bandaid_df):,}")
    col4.metric("Bandaid Failure Rate",
                pct(len(bandaid_fail_df) / max(len(bandaid_df), 1)))

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fraud Scenario Mix")
        fc = fraud_df["scenario"].value_counts().reset_index()
        fc.columns = ["scenario", "count"]
        fig = px.pie(fc, names="scenario", values="count",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(
            paper_bgcolor=PALETTE["surface"], font_color=PALETTE["text"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Top 20 Reps by Fraud Involvement")
        top_fraud = rep_filtered.nlargest(20, "fraud_involvement")[
            ["rep_name","team","total_calls","fraud_involvement","avg_gaming"]
        ]
        fig2 = px.bar(
            top_fraud.sort_values("fraud_involvement"),
            x="fraud_involvement", y="rep_name", orientation="h",
            color="avg_gaming",
            color_continuous_scale=["#4ECDC4","#FF4C61"],
            labels={"fraud_involvement":"Fraud Involvement","rep_name":"Rep"},
        )
        fig2.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"], height=480,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Bandaid failure by rep
    st.subheader("Top 20 Reps: Bandaid Failure Rate")
    top_bandaid = rep_filtered[rep_filtered["bandaid_count"] > 0].nlargest(20, "bandaid_fail_rate")
    fig3 = px.bar(
        top_bandaid.sort_values("bandaid_fail_rate"),
        x="bandaid_fail_rate", y="rep_name", orientation="h",
        color="bandaid_fail_rate",
        color_continuous_scale=["#FFB347","#FF4C61"],
        labels={"bandaid_fail_rate":"Bandaid Failure Rate","rep_name":"Rep"},
    )
    fig3.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], xaxis_tickformat=".0%", height=480,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Unauthorized credits table
    st.subheader("Unauthorized Bandaid Credits (Sample)")
    unauth = df_f[
        (df_f["credit_type"] == "bandaid") &
        (df_f["credit_authorized"] == False)
    ][["call_date","rep_name","team","customer_id","credit_amount",
       "repeat_contact_31_60d","scenario"]].head(100)
    st.dataframe(unauth, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: GOVERNANCE SIGNALS (DAR, DRL, DOV, POR, SII)
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Governance Signals":
    st.markdown("# Governance Signals")
    st.caption("DAR · DRL · DOV · POR · SII — from *When KPIs Lie*")

    # ── DAR: Deferred Action Rate ──────────────────────────────────────────
    # repeat_contact_31_60d WHERE resolution_flag=True / total resolution_flag=True
    res_true_calls = df_f[df_f["resolution_flag"] == True]
    DAR = bool_mean(res_true_calls["repeat_contact_31_60d"]) if len(res_true_calls) > 0 else 0.0

    # ── DOV: Degradation of Validity ──────────────────────────────────────
    # Accuracy of resolution_flag predicting true_resolution over time
    monthly_gov = []
    for m in sorted(df_f["month"].unique()):
        mslice = df_f[df_f["month"] == m]
        if len(mslice) < 10:
            continue
        proxy = mslice["resolution_flag"].astype(float)
        truth = mslice["true_resolution"].astype(float)
        agree = (proxy.fillna(0) == truth.fillna(0)).mean()
        gap   = bool_mean(proxy) - bool_mean(truth)
        dar_m = bool_mean(mslice[mslice["resolution_flag"] == True]["repeat_contact_31_60d"])
        monthly_gov.append({
            "month": m,
            "proxy_accuracy": agree,
            "resolution_gap": gap,
            "DAR": dar_m,
            "proxy_rate": bool_mean(proxy),
            "true_rate": bool_mean(truth),
        })
    gov_df = pd.DataFrame(monthly_gov)

    DOV = 0.0
    if len(gov_df) >= 2:
        DOV = gov_df["proxy_accuracy"].iloc[0] - gov_df["proxy_accuracy"].iloc[-1]

    # ── POR: Proxy Overshoot Ratio ─────────────────────────────────────────
    POR = 0.0
    if len(gov_df) >= 2:
        delta_proxy = gov_df["proxy_rate"].iloc[-1] - gov_df["proxy_rate"].iloc[0]
        delta_true  = gov_df["true_rate"].iloc[-1]  - gov_df["true_rate"].iloc[0]
        POR = (delta_proxy / delta_true) if abs(delta_true) > 0.001 else 0.0

    # ── DRL: Distribution Replication Loss (simplified JS-like divergence) ─
    early = df_f[df_f["month"] <= gov_df["month"].iloc[len(gov_df)//2]] if len(gov_df) > 1 else df_f
    late  = df_f[df_f["month"] >  gov_df["month"].iloc[len(gov_df)//2]] if len(gov_df) > 1 else df_f
    def scenario_dist(d):
        counts = d["scenario"].value_counts(normalize=True)
        return counts
    p = scenario_dist(early[early["resolution_flag"] == True])
    q = scenario_dist(late[late["resolution_flag"] == True])
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=1e-9)
    q = q.reindex(idx, fill_value=1e-9)
    m_dist = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m_dist)) + 0.5 * np.sum(q * np.log(q / m_dist)))
    DRL = js

    # ── SII composite ─────────────────────────────────────────────────────
    DAR_n  = min(DAR, 1.0)
    DRL_n  = min(DRL, 1.0)
    DOV_n  = min(max(DOV, 0.0), 1.0)
    POR_n  = min(max(POR - 1.0, 0.0) / 2.0, 1.0) if POR > 1 else 0.0
    SII    = 100 * (0.30*DAR_n + 0.20*DRL_n + 0.25*DOV_n + 0.25*POR_n)

    # Display
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("DAR",  f"{DAR:.4f}",  help="Deferred Action Rate: repeat_31_60d | resolution_flag=True")
    gc2.metric("DRL",  f"{DRL:.4f}",  help="Distribution Replication Loss: JS divergence of post-success scenario mix")
    gc3.metric("DOV",  f"{DOV:.4f}",  help="Degradation of Validity: decay in proxy-truth accuracy")
    gc4.metric("POR",  f"{POR:.4f}",  help="Proxy Overshoot Ratio: Δproxy / Δtrue resolution")
    gc5.metric("SII",  f"{SII:.2f}",  help="System Integrity Index (0–100). Higher = more drift signal.")

    if SII >= 60:
        st.error(f"⚠️ SII = {SII:.1f} — High governance risk detected.")
    elif SII >= 30:
        st.warning(f"⚡ SII = {SII:.1f} — Moderate drift signal. Monitor closely.")
    else:
        st.success(f"✅ SII = {SII:.1f} — Within acceptable range.")

    st.markdown("---")

    if len(gov_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Proxy Accuracy Decay (DOV)")
            fig = px.line(
                gov_df, x="month", y="proxy_accuracy",
                markers=True,
                labels={"proxy_accuracy": "Proxy→True Accuracy", "month": "Month"},
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig.update_layout(
                plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
                font_color=PALETTE["text"], yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("DAR Monthly Trend")
            fig2 = px.bar(
                gov_df, x="month", y="DAR",
                color="DAR",
                color_continuous_scale=["#4ECDC4","#FF4C61"],
                labels={"DAR": "Deferred Action Rate", "month": "Month"},
            )
            fig2.update_layout(
                plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
                font_color=PALETTE["text"], coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Resolution Gap Monthly (POR context)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=gov_df["month"], y=gov_df["proxy_rate"],
            name="Proxy Rate", line=dict(color=PALETTE["primary"], width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=gov_df["month"], y=gov_df["true_rate"],
            name="True Rate", line=dict(color=PALETTE["danger"], width=2, dash="dash"),
        ))
        fig3.add_trace(go.Scatter(
            x=gov_df["month"], y=gov_df["resolution_gap"],
            name="Gap", fill="tozeroy",
            line=dict(color=PALETTE["warning"], width=1),
            fillcolor="rgba(255,179,71,0.12)",
        ))
        fig3.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["text"], yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Per-team SII approximation
    st.subheader("Governance Risk by Team (DAR + Resolution Gap proxy)")
    team_gov = team_metrics[["team","resolution_gap","avg_gaming","bandaid_fail_rate","fraud_involvement"]].copy()
    team_gov["risk_score"] = (
        0.35 * team_gov["resolution_gap"].clip(0) +
        0.25 * team_gov["avg_gaming"] +
        0.20 * team_gov["bandaid_fail_rate"] +
        0.20 * team_gov["fraud_involvement"]
    ) * 100
    fig4 = px.bar(
        team_gov.sort_values("risk_score", ascending=False),
        x="team", y="risk_score",
        color="risk_score",
        color_continuous_scale=["#4ECDC4","#FFB347","#FF4C61"],
        labels={"risk_score": "Governance Risk Score", "team": "Team"},
    )
    fig4.update_layout(
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"], coloraxis_showscale=False,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig4, use_container_width=True)
