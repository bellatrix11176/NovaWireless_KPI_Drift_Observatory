"""
NovaWireless KPI Drift Observatory — Paper Figure Generator
============================================================
Proves Goodhart's Law in action: metrics that look green while reality decays.

Output: output/figures/*.png  (300 DPI, publication-ready)
Run:    python src/generate_paper_figures.py

Figures produced:
  Fig 1 — The Green Dashboard Lie      (proxy FCR vs true FCR, side by side)
  Fig 2 — The Scenario Contradiction   (gamed_metric: 98% proxy, 14% true)
  Fig 3 — Trust Decay vs KPI Lift      (monthly: proxy climbs, trust falls)
  Fig 4 — The Bandaid Economy          (unauthorized credits + repeat contacts)
  Fig 5 — Gaming Propensity Drift      (rep gaming drift over 12 months)
  Fig 6 — The Goodhart Summary Panel   (all signals in one publication figure)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    "#0D1117",
    "figure.facecolor":  "#0D1117",
    "text.color":        "#E6EDF3",
    "axes.labelcolor":   "#E6EDF3",
    "xtick.color":       "#8892A4",
    "ytick.color":       "#8892A4",
    "axes.edgecolor":    "#21262D",
    "grid.color":        "#21262D",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#21262D",
})

# ── COLORS ────────────────────────────────────────────────────────────────────
C_GREEN   = "#00C2CB"   # looks-good metrics
C_RED     = "#FF4C61"   # reality metrics
C_WARN    = "#FFB347"   # warning / gap
C_NEUTRAL = "#8892A4"   # secondary
C_TRUST   = "#A78BFA"   # trust / purple
C_GAMING  = "#F97316"   # gaming propensity
C_BAND    = "#EC4899"   # bandaid

pct_fmt  = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
pct_fmt1 = FuncFormatter(lambda x, _: f"{x*100:.1f}%")

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "external")
OUT_DIR      = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data(data_dir):
    pattern = os.path.join(data_dir, "calls_sanitized_2025-*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found at: {pattern}")
    print(f"Loading {len(files)} file(s)…")
    frames = []
    for f in files:
        tmp = pd.read_csv(f, low_memory=False)
        tmp["source_file"] = os.path.basename(f)
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    df["month"]     = df["call_date"].dt.to_period("M").astype(str)
    df["month_num"] = df["call_date"].dt.month

    bool_cols = ["true_resolution","resolution_flag","repeat_contact_30d",
                 "repeat_contact_31_60d","escalation_flag","credit_applied",
                 "credit_authorized","customer_is_churned","is_repeat_call"]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip().str.lower().map(
                    {"true":True,"false":False,"1":True,"0":False}
                )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["aht_secs","rep_gaming_propensity","rep_burnout_level",
                "rep_policy_skill","customer_trust_baseline",
                "customer_churn_risk_effective","agent_qa_score","credit_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived
    df["is_fcr_true"]  = (
        (df["true_resolution"]      == 1) &
        (df["repeat_contact_30d"]   == 0) &
        (df["repeat_contact_31_60d"]== 0)
    ).astype(int)
    df["is_fcr_proxy"] = (
        (df["resolution_flag"]       == 1) &
        (df["repeat_contact_30d"]   == 0) &
        (df["repeat_contact_31_60d"]== 0)
    ).astype(int)
    df["is_bandaid_fail"] = (
        (df["credit_type"]           == "bandaid") &
        (df["repeat_contact_31_60d"] == 1)
    ).astype(int)
    return df

df = load_data(DATA_DIR)
months = sorted(df["month"].unique())
n_months = len(months)
print(f"Loaded {len(df):,} rows across {n_months} month(s): {months[0]} → {months[-1]}")

FRAUD = {"fraud_store_promo","fraud_line_add","fraud_hic_exchange","fraud_care_promo"}


# ── MONTHLY AGGREGATES ────────────────────────────────────────────────────────
monthly = (
    df.groupby(["month","month_num"])
    .agg(
        proxy_fcr     = ("is_fcr_proxy",             "mean"),
        true_fcr      = ("is_fcr_true",              "mean"),
        proxy_res     = ("resolution_flag",           "mean"),
        true_res      = ("true_resolution",           "mean"),
        avg_trust     = ("customer_trust_baseline",   "mean"),
        avg_gaming    = ("rep_gaming_propensity",     "mean"),
        avg_burnout   = ("rep_burnout_level",         "mean"),
        repeat_30d    = ("repeat_contact_30d",        "mean"),
        repeat_31_60d = ("repeat_contact_31_60d",     "mean"),
        bandaid_rate  = ("credit_type",               lambda x: (x=="bandaid").mean()),
        bandaid_fail  = ("is_bandaid_fail",           "mean"),
        churn_rate    = ("customer_is_churned",       "mean"),
        avg_aht       = ("aht_secs",                  "mean"),
    )
    .reset_index()
    .sort_values("month_num")
)

# ── SCENARIO AGGREGATES ───────────────────────────────────────────────────────
scenarios = (
    df.groupby("scenario")
    .agg(
        proxy_res     = ("resolution_flag",         "mean"),
        true_res      = ("true_resolution",         "mean"),
        proxy_fcr     = ("is_fcr_proxy",            "mean"),
        true_fcr      = ("is_fcr_true",             "mean"),
        avg_trust     = ("customer_trust_baseline", "mean"),
        repeat_31_60d = ("repeat_contact_31_60d",   "mean"),
        bandaid_rate  = ("credit_type",             lambda x: (x=="bandaid").mean()),
        call_count    = ("resolution_flag",         "count"),
    )
    .reset_index()
)
scenarios["gap"] = scenarios["proxy_res"] - scenarios["true_res"]
scenarios = scenarios.sort_values("gap", ascending=False)


def save(fig, name, dpi=300):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — THE GREEN DASHBOARD LIE
# Proxy FCR vs True FCR: what management sees vs what is real
# ══════════════════════════════════════════════════════════════════════════════
def fig1_green_lie():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figure 1 — The Green Dashboard Lie\n"
        "What management sees vs. what is actually happening",
        fontsize=13, fontweight="bold", color="#E6EDF3", y=1.02
    )

    # Left: FCR comparison bars
    ax = axes[0]
    labels  = ["Proxy FCR\n(What KPI shows)", "True FCR\n(What actually happened)"]
    vals    = [monthly["proxy_fcr"].mean(), monthly["true_fcr"].mean()]
    colors  = [C_GREEN, C_RED]
    bars    = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=bar.get_facecolor())
    gap = vals[0] - vals[1]
    ax.annotate(
        f"Gap: {gap*100:.1f}pp\n← Goodhart's Law",
        xy=(0.5, (vals[0]+vals[1])/2),
        xycoords=("data","data"),
        ha="center", fontsize=10, color=C_WARN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D", edgecolor=C_WARN, alpha=0.9)
    )
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_ylim(0, 0.85)
    ax.set_title("First Contact Resolution: Proxy vs Truth", pad=12)
    ax.set_ylabel("Rate")

    # Right: Resolution rate gap (proxy vs true)
    ax2 = axes[1]
    x   = np.arange(len(months)) if n_months > 1 else [0]
    ax2.plot(x, monthly["proxy_res"], color=C_GREEN, linewidth=2.5,
             marker="o", markersize=5, label="Proxy Resolution (KPI)")
    ax2.plot(x, monthly["true_res"],  color=C_RED,   linewidth=2.5,
             marker="s", markersize=5, linestyle="--", label="True Resolution (Reality)")
    ax2.fill_between(x, monthly["proxy_res"], monthly["true_res"],
                     alpha=0.15, color=C_WARN, label="Gap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha="right")
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.set_title("Resolution Rate Drift Over Time", pad=12)
    ax2.set_ylabel("Resolution Rate")
    ax2.legend()
    if n_months == 1:
        ax2.text(0.5, 0.5, "Load all 12 months\nfor full drift view",
                 transform=ax2.transAxes, ha="center", va="center",
                 color=C_NEUTRAL, fontsize=10)

    fig.tight_layout()
    save(fig, "fig1_green_dashboard_lie.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — THE SCENARIO CONTRADICTION
# gamed_metric: proxy=98%, true=14%. Clean: proxy=98%, true=89%.
# Same proxy. Completely different reality.
# ══════════════════════════════════════════════════════════════════════════════
def fig2_scenario_contradiction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 2 — The Scenario Contradiction\n"
        "Same proxy resolution rate. Completely different reality.",
        fontsize=13, fontweight="bold", color="#E6EDF3", y=1.02
    )

    scen_order = scenarios["scenario"].tolist()
    x = np.arange(len(scen_order))
    w = 0.35

    # Left: proxy vs true resolution by scenario
    ax = axes[0]
    ax.barh(x + w/2, scenarios["proxy_res"], w, color=C_GREEN, label="Proxy Resolution", zorder=3)
    ax.barh(x - w/2, scenarios["true_res"],  w, color=C_RED,   label="True Resolution",  zorder=3)
    ax.set_yticks(x)
    ax.set_yticklabels(scen_order, fontsize=8)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_title("Proxy vs True Resolution by Scenario", pad=12)
    ax.set_xlabel("Resolution Rate")
    ax.legend()
    # Annotate the worst offender
    gamed_idx = scen_order.index("gamed_metric") if "gamed_metric" in scen_order else 0
    ax.axhline(gamed_idx, color=C_WARN, linewidth=1, linestyle=":", alpha=0.7)
    ax.text(0.97, gamed_idx + 0.4, "← gamed_metric",
            transform=ax.get_yaxis_transform(), ha="right",
            color=C_WARN, fontsize=8, fontweight="bold")

    # Right: resolution gap by scenario with repeat contact overlay
    ax2 = axes[1]
    gap_colors = [C_RED if g > 0.5 else C_WARN if g > 0.2 else C_GREEN
                  for g in scenarios["gap"]]
    bars = ax2.barh(x, scenarios["gap"], color=gap_colors, zorder=3)
    ax2.set_yticks(x)
    ax2.set_yticklabels(scen_order, fontsize=8)
    ax2.xaxis.set_major_formatter(pct_fmt)
    ax2.set_title("Resolution Gap (Proxy − True)\n+ Repeat Contact 31–60d", pad=12)
    ax2.set_xlabel("Gap (percentage points)")

    # Overlay repeat contact as dots
    ax2b = ax2.twiny()
    ax2b.scatter(scenarios["repeat_31_60d"], x, color=C_WARN, zorder=5,
                 s=60, marker="D", label="Repeat 31–60d")
    ax2b.xaxis.set_major_formatter(pct_fmt)
    ax2b.set_xlabel("Repeat Contact Rate (31–60d)", color=C_WARN)
    ax2b.tick_params(colors=C_WARN)

    legend_els = [
        mpatches.Patch(color=C_RED,   label="Gap > 50pp"),
        mpatches.Patch(color=C_WARN,  label="Gap 20–50pp"),
        mpatches.Patch(color=C_GREEN, label="Gap < 20pp"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=C_WARN,
               markersize=7, label="Repeat 31–60d"),
    ]
    ax2.legend(handles=legend_els, loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_scenario_contradiction.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — TRUST DECAY vs KPI LIFT
# The proxy climbs. Trust erodes. AHT stays flat. The system is blind.
# ══════════════════════════════════════════════════════════════════════════════
def fig3_trust_decay():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Figure 3 — KPI Lift vs Reality Decay\n"
        "Green metrics climb. Customer trust erodes. The system is blind.",
        fontsize=13, fontweight="bold", color="#E6EDF3", y=1.01
    )

    x      = np.arange(len(months))
    xticks = months

    def base_ax(ax, title, ylabel):
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        if n_months == 1:
            ax.text(0.5, 0.5, "Load all 12 months\nfor drift view",
                    transform=ax.transAxes, ha="center", va="center",
                    color=C_NEUTRAL, fontsize=9)

    # Top-left: Proxy FCR (green, climbing) vs Trust (purple, falling)
    ax = axes[0][0]
    ax.set_facecolor("#0D1117")
    ln1 = ax.plot(x, monthly["proxy_fcr"], color=C_GREEN, linewidth=2.5,
                  marker="o", markersize=4, label="Proxy FCR (KPI — looks good)")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax2 = ax.twinx()
    ln2 = ax2.plot(x, monthly["avg_trust"], color=C_TRUST, linewidth=2.5,
                   marker="s", markersize=4, linestyle="--", label="Avg Customer Trust")
    ax2.set_ylabel("Customer Trust Score", color=C_TRUST)
    ax2.tick_params(colors=C_TRUST)
    lines = ln1 + ln2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7)
    base_ax(ax, "Proxy FCR (↑ Green) vs Customer Trust (↓ Decaying)", "Proxy FCR Rate")

    # Top-right: AHT over time (management loves low AHT)
    ax = axes[0][1]
    ax.plot(x, monthly["avg_aht"], color=C_GREEN, linewidth=2.5,
            marker="o", markersize=4, label="Avg AHT (secs)")
    ax.fill_between(x, monthly["avg_aht"].min(), monthly["avg_aht"],
                    alpha=0.1, color=C_GREEN)
    ax.set_ylabel("Seconds")
    base_ax(ax, "AHT Over Time\n(Management KPI — lower = better bonus)", "AHT (secs)")
    ax.legend()

    # Bottom-left: Repeat contacts (the hidden cost of short calls)
    ax = axes[1][0]
    ax.plot(x, monthly["repeat_30d"],    color=C_WARN, linewidth=2,
            marker="o", markersize=4, label="Repeat Contact 30d")
    ax.plot(x, monthly["repeat_31_60d"], color=C_RED,  linewidth=2,
            marker="s", markersize=4, linestyle="--", label="Repeat Contact 31–60d")
    ax.fill_between(x, 0, monthly["repeat_31_60d"], alpha=0.12, color=C_RED)
    ax.yaxis.set_major_formatter(pct_fmt)
    base_ax(ax, "Repeat Contacts Over Time\n(The hidden cost — not on any KPI dashboard)", "Rate")
    ax.legend()

    # Bottom-right: Gaming propensity drift
    ax = axes[1][1]
    ax.plot(x, monthly["avg_gaming"], color=C_GAMING, linewidth=2.5,
            marker="o", markersize=4, label="Avg Gaming Propensity")
    ax.plot(x, monthly["avg_burnout"], color=C_RED, linewidth=2,
            marker="s", markersize=4, linestyle=":", label="Avg Burnout")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.2f}"))
    base_ax(ax, "Rep Gaming Propensity & Burnout Drift\n(Reps learn the system — proxy KPIs reward it)", "Score")
    ax.legend()

    fig.tight_layout()
    save(fig, "fig3_trust_decay_vs_kpi_lift.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — THE BANDAID ECONOMY
# Unauthorized hush-money credits that make calls look resolved.
# The repeat contact proves the issue wasn't actually fixed.
# ══════════════════════════════════════════════════════════════════════════════
def fig4_bandaid_economy():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Figure 4 — The Bandaid Economy\n"
        "Unauthorized credits silence customers. Issues resurface 31–60 days later.",
        fontsize=13, fontweight="bold", color="#E6EDF3", y=1.02
    )

    # Left: credit type breakdown
    ax = axes[0]
    credit_counts = df["credit_type"].value_counts()
    colors_map = {
        "bandaid":       C_RED,
        "none":          C_NEUTRAL,
        "dispute_credit":C_GREEN,
        "service_credit":C_GREEN,
        "courtesy":      C_WARN,
        "fee_waiver":    C_WARN,
    }
    wedge_colors = [colors_map.get(c, C_NEUTRAL) for c in credit_counts.index]
    wedges, texts, autotexts = ax.pie(
        credit_counts.values,
        labels=credit_counts.index,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color":"#E6EDF3","fontsize":8},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title("Credit Type Distribution\n(Bandaid = unauthorized hush money)", pad=12)

    # Middle: bandaid rate vs bandaid failure rate monthly
    ax2 = axes[1]
    x = np.arange(len(months))
    ax2.bar(x - 0.2, monthly["bandaid_rate"],  0.35, color=C_BAND,  label="Bandaid Rate",        zorder=3)
    ax2.bar(x + 0.2, monthly["bandaid_fail"],  0.35, color=C_RED,   label="Bandaid Failure Rate", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.set_title("Bandaid Rate vs Failure Rate\n(Failure = customer called back 31–60d later)", pad=12)
    ax2.set_ylabel("Rate")
    ax2.legend()
    if n_months == 1:
        ax2.text(0.5, 0.5, "Load all 12 months\nfor trend view",
                 transform=ax2.transAxes, ha="center", va="center",
                 color=C_NEUTRAL, fontsize=9)

    # Right: the core fraud signal scatter
    # x = bandaid rate per rep, y = repeat_31_60d rate per rep
    rep_agg = df.groupby("rep_id").agg(
        bandaid_rate  = ("credit_type",             lambda x: (x=="bandaid").mean()),
        repeat_31_60d = ("repeat_contact_31_60d",   "mean"),
        gaming        = ("rep_gaming_propensity",   "mean"),
        true_fcr      = ("is_fcr_true",             "mean"),
    ).reset_index()

    ax3 = axes[2]
    sc = ax3.scatter(
        rep_agg["bandaid_rate"], rep_agg["repeat_31_60d"],
        c=rep_agg["gaming"], cmap="RdYlGn_r",
        s=40, alpha=0.7, zorder=3,
    )
    plt.colorbar(sc, ax=ax3, label="Gaming Propensity")
    # Trend line
    z = np.polyfit(rep_agg["bandaid_rate"].fillna(0),
                   rep_agg["repeat_31_60d"].fillna(0), 1)
    p = np.poly1d(z)
    xr = np.linspace(rep_agg["bandaid_rate"].min(), rep_agg["bandaid_rate"].max(), 100)
    ax3.plot(xr, p(xr), color=C_RED, linewidth=1.5, linestyle="--", label="Trend")
    ax3.xaxis.set_major_formatter(pct_fmt)
    ax3.yaxis.set_major_formatter(pct_fmt)
    ax3.set_xlabel("Rep Bandaid Rate")
    ax3.set_ylabel("Rep Repeat Contact Rate (31–60d)")
    ax3.set_title("Bandaid Rate vs Repeat Contacts\n(Color = gaming propensity)", pad=12)
    ax3.legend(fontsize=7)
    # Correlation annotation
    corr = rep_agg["bandaid_rate"].corr(rep_agg["repeat_31_60d"])
    ax3.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax3.transAxes,
             color=C_WARN, fontsize=9, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig4_bandaid_economy.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — GAMING PROPENSITY DRIFT
# Reps learn the system. The metric that rewards gaming stays green.
# ══════════════════════════════════════════════════════════════════════════════
def fig5_gaming_drift():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 5 — Gaming Propensity Drift\n"
        "Reps learn what the system rewards. Proxy KPIs never flag it.",
        fontsize=13, fontweight="bold", color="#E6EDF3", y=1.02
    )

    # Left: distribution of gaming propensity across all reps
    ax = axes[0]
    rep_gaming = df.groupby("rep_id")["rep_gaming_propensity"].mean().sort_values()
    threshold  = 0.5
    low  = rep_gaming[rep_gaming < threshold]
    high = rep_gaming[rep_gaming >= threshold]
    ax.barh(range(len(low)),  low.values,  color=C_GREEN,  alpha=0.8, label=f"Gaming < {threshold} ({len(low)} reps)")
    ax.barh(range(len(low), len(rep_gaming)), high.values, color=C_RED, alpha=0.8, label=f"Gaming ≥ {threshold} ({len(high)} reps)")
    ax.axvline(threshold, color=C_WARN, linewidth=1.5, linestyle="--", label=f"Threshold = {threshold}")
    ax.set_xlabel("Avg Gaming Propensity")
    ax.set_ylabel("Reps (sorted)")
    ax.set_title(f"Gaming Propensity Distribution\n({len(rep_gaming)} reps total)", pad=12)
    ax.legend(fontsize=7)
    ax.set_yticks([])

    # Right: gaming vs resolution gap scatter with FCR overlay
    ax2 = axes[1]
    rep_agg = df.groupby("rep_id").agg(
        gaming     = ("rep_gaming_propensity", "mean"),
        gap        = ("resolution_flag",
                      lambda x: x.astype(float).mean() - df.loc[x.index,"true_resolution"].astype(float).mean()),
        true_fcr   = ("is_fcr_true",   "mean"),
        proxy_fcr  = ("is_fcr_proxy",  "mean"),
        burnout    = ("rep_burnout_level", "mean"),
    ).reset_index()

    sc = ax2.scatter(
        rep_agg["gaming"], rep_agg["gap"],
        c=rep_agg["true_fcr"], cmap="RdYlGn",
        s=45, alpha=0.75, zorder=3,
    )
    plt.colorbar(sc, ax=ax2, label="True FCR")

    # Trend
    z = np.polyfit(rep_agg["gaming"].fillna(0), rep_agg["gap"].fillna(0), 1)
    p = np.poly1d(z)
    xr = np.linspace(rep_agg["gaming"].min(), rep_agg["gaming"].max(), 100)
    ax2.plot(xr, p(xr), color=C_RED, linewidth=1.5, linestyle="--")

    corr = rep_agg["gaming"].corr(rep_agg["gap"])
    ax2.text(0.05, 0.92, f"r = {corr:.3f}  (gaming ↔ resolution gap)",
             transform=ax2.transAxes, color=C_WARN, fontsize=9, fontweight="bold")

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.2f}"))
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.set_xlabel("Avg Gaming Propensity")
    ax2.set_ylabel("Resolution Gap (Proxy − True)")
    ax2.set_title("Gaming Propensity vs Resolution Gap\n(Color = True FCR — green is good, red is bad)", pad=12)

    fig.tight_layout()
    save(fig, "fig5_gaming_drift.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — THE GOODHART SUMMARY PANEL
# One figure. The whole argument. For the paper abstract / opening figure.
# ══════════════════════════════════════════════════════════════════════════════
def fig6_summary_panel():
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "NovaWireless KPI Drift Observatory — Goodhart's Law in Action\n"
        "\"When a measure becomes a target, it ceases to be a good measure.\" — Goodhart, 1975",
        fontsize=14, fontweight="bold", color="#E6EDF3", y=1.01
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: FCR gap big numbers ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    proxy_fcr_val = monthly["proxy_fcr"].mean()
    true_fcr_val  = monthly["true_fcr"].mean()
    ax_a.bar(["Proxy FCR\n(KPI)"], [proxy_fcr_val], color=C_GREEN, width=0.4, zorder=3)
    ax_a.bar(["True FCR\n(Reality)"], [true_fcr_val], color=C_RED, width=0.4, zorder=3)
    ax_a.set_ylim(0, 0.85)
    ax_a.yaxis.set_major_formatter(pct_fmt)
    gap = proxy_fcr_val - true_fcr_val
    ax_a.set_title(f"A — FCR Gap: {gap*100:.1f}pp\nProxy vs Truth", pad=8)
    for label, val, color in zip(["Proxy FCR\n(KPI)","True FCR\n(Reality)"],
                                  [proxy_fcr_val, true_fcr_val], [C_GREEN,C_RED]):
        ax_a.text(["Proxy FCR\n(KPI)","True FCR\n(Reality)"].index(label),
                  val + 0.02, f"{val*100:.1f}%",
                  ha="center", fontsize=12, fontweight="bold", color=color)

    # ── Panel B: Proxy vs True by scenario (top 5 worst gaps) ────────────
    ax_b = fig.add_subplot(gs[0, 1])
    top5 = scenarios.head(5)
    y    = np.arange(len(top5))
    ax_b.barh(y + 0.2, top5["proxy_res"], 0.35, color=C_GREEN, label="Proxy", zorder=3)
    ax_b.barh(y - 0.2, top5["true_res"],  0.35, color=C_RED,   label="True",  zorder=3)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(top5["scenario"], fontsize=7)
    ax_b.xaxis.set_major_formatter(pct_fmt)
    ax_b.set_title("B — Top 5 Worst Gaps\nBy Scenario", pad=8)
    ax_b.legend(fontsize=7)

    # ── Panel C: Trust decay vs proxy lift ────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    x = np.arange(len(months))
    ax_c.plot(x, monthly["proxy_fcr"], color=C_GREEN, linewidth=2,
              marker="o", markersize=3, label="Proxy FCR")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x, monthly["avg_trust"], color=C_TRUST, linewidth=2,
               marker="s", markersize=3, linestyle="--", label="Customer Trust")
    ax_c2.set_ylabel("Trust Score", color=C_TRUST, fontsize=7)
    ax_c2.tick_params(colors=C_TRUST, labelsize=6)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(months, rotation=45, ha="right", fontsize=6)
    ax_c.yaxis.set_major_formatter(pct_fmt)
    ax_c.set_title("C — KPI Lifts, Trust Decays", pad=8)
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1+lines2, labels1+labels2, fontsize=6, loc="upper left")
    if n_months == 1:
        ax_c.text(0.5, 0.5, "12 months\nneeded", transform=ax_c.transAxes,
                  ha="center", va="center", color=C_NEUTRAL, fontsize=9)

    # ── Panel D: Repeat contacts (the hidden tax) ─────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.bar(x - 0.2, monthly["repeat_30d"],    0.35, color=C_WARN, label="30d",    zorder=3)
    ax_d.bar(x + 0.2, monthly["repeat_31_60d"], 0.35, color=C_RED,  label="31–60d", zorder=3)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(months, rotation=45, ha="right", fontsize=6)
    ax_d.yaxis.set_major_formatter(pct_fmt)
    ax_d.set_title("D — Repeat Contacts\n(Hidden cost, not on KPI dashboard)", pad=8)
    ax_d.legend(fontsize=7)

    # ── Panel E: Bandaid fail correlation ─────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    rep_agg = df.groupby("rep_id").agg(
        bandaid_rate  = ("credit_type",           lambda x: (x=="bandaid").mean()),
        repeat_31_60d = ("repeat_contact_31_60d", "mean"),
        gaming        = ("rep_gaming_propensity", "mean"),
    ).reset_index()
    sc = ax_e.scatter(rep_agg["bandaid_rate"], rep_agg["repeat_31_60d"],
                      c=rep_agg["gaming"], cmap="RdYlGn_r",
                      s=25, alpha=0.6, zorder=3)
    z  = np.polyfit(rep_agg["bandaid_rate"].fillna(0), rep_agg["repeat_31_60d"].fillna(0), 1)
    p  = np.poly1d(z)
    xr = np.linspace(rep_agg["bandaid_rate"].min(), rep_agg["bandaid_rate"].max(), 100)
    ax_e.plot(xr, p(xr), color=C_RED, linewidth=1.5, linestyle="--")
    corr = rep_agg["bandaid_rate"].corr(rep_agg["repeat_31_60d"])
    ax_e.text(0.05, 0.9, f"r = {corr:.3f}", transform=ax_e.transAxes,
              color=C_WARN, fontsize=9, fontweight="bold")
    ax_e.xaxis.set_major_formatter(pct_fmt)
    ax_e.yaxis.set_major_formatter(pct_fmt)
    ax_e.set_xlabel("Bandaid Rate", fontsize=7)
    ax_e.set_ylabel("Repeat 31–60d", fontsize=7)
    ax_e.set_title("E — Bandaid → Repeat Contact\n(Hush money doesn't fix problems)", pad=8)

    # ── Panel F: Gaming vs Gap ────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    rep_agg2 = df.groupby("rep_id").agg(
        gaming = ("rep_gaming_propensity", "mean"),
        gap    = ("resolution_flag",
                  lambda x: x.astype(float).mean() - df.loc[x.index,"true_resolution"].astype(float).mean()),
    ).reset_index()
    ax_f.scatter(rep_agg2["gaming"], rep_agg2["gap"],
                 color=C_GAMING, s=25, alpha=0.6, zorder=3)
    z2 = np.polyfit(rep_agg2["gaming"].fillna(0), rep_agg2["gap"].fillna(0), 1)
    p2 = np.poly1d(z2)
    xr2 = np.linspace(rep_agg2["gaming"].min(), rep_agg2["gaming"].max(), 100)
    ax_f.plot(xr2, p2(xr2), color=C_RED, linewidth=1.5, linestyle="--")
    corr2 = rep_agg2["gaming"].corr(rep_agg2["gap"])
    ax_f.text(0.05, 0.9, f"r = {corr2:.3f}", transform=ax_f.transAxes,
              color=C_WARN, fontsize=9, fontweight="bold")
    ax_f.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.2f}"))
    ax_f.yaxis.set_major_formatter(pct_fmt)
    ax_f.set_xlabel("Gaming Propensity", fontsize=7)
    ax_f.set_ylabel("Resolution Gap", fontsize=7)
    ax_f.set_title("F — Gaming ↔ Resolution Gap\n(More gaming = bigger KPI lie)", pad=8)

    save(fig, "fig6_goodhart_summary_panel.png", dpi=300)


# ── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating paper figures…\n")
    fig1_green_lie()
    fig2_scenario_contradiction()
    fig3_trust_decay()
    fig4_bandaid_economy()
    fig5_gaming_drift()
    fig6_summary_panel()
    print(f"\nAll figures saved to: {OUT_DIR}")
    print("300 DPI PNGs — publication ready.")
