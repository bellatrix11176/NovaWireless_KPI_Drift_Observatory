# NovaWireless KPI Drift Observatory

### Your resolution rate is 89%. Your actual fix rate is 47%. This is the repo that proves it.

---

⚠️ Resource Notice
This pipeline is designed to be run iteratively on small datasets. Do not run against large datasets in a single pass. Process one month at a time to avoid memory and CPU overload. Adjust chunk sizes based on your available system resources.

---

Every contact center executive has seen the dashboard: resolution rates climbing, handle times improving, customer satisfaction holding steady. The numbers say the operation is getting better.

The numbers are wrong.

When agents are optimized against a proxy KPI — and especially when AI systems accelerate that optimization — the proxy diverges from the outcome it was supposed to measure. Resolution rates go up because the system records calls as resolved, not because problems are actually being fixed. The 30-day FCR window stays clean because bandaid credits suppress repeat contacts just long enough. And the dashboard never shows you the 31–60 day spike where the real failures surface.

This repository computes the **System Integrity Index (SII)** — a composite governance signal that tells you whether your measurement environment is still trustworthy enough to optimize against. When it isn't, the SII vetoes further AI optimization until labels and training targets are re-audited.

If you're using AI to route calls, score agents, or predict churn, and you're not measuring measurement integrity, you're optimizing against a target that's already broken.

---

## The 42-Point Gap No Dashboard Shows You

Across 82,305 synthetic calls over 12 months:

| What the Dashboard Says | What Actually Happened | The Gap |
|---|---|---|
| 89.4% resolution rate | 47.3% true resolution | **42.1 percentage points** |

The proxy improved nearly **2× faster** than the true outcome (POR = 1.89). That means every month the AI optimizes against this label, it's training on data that's further from reality than the month before.

The resolution flag has an odds ratio of **0.99** against churn (95% CI 0.94–1.05, p = .78). Statistically indistinguishable from a coin flip. The Terminal Exit Rate for proxy-resolved calls (27.6%) is identical to the baseline churn rate (27.63%).

Your highest-confidence label carries zero predictive information for the outcome it's supposed to prevent.

---

## What the SII Measures

The System Integrity Index is composed of four normalized signals, each targeting a different failure mode:

| Component | Weight | What It Catches |
|---|---|---|
| **DAR** — Deferred Action Rate | 0.30 | Calls labeled "resolved" that generate repeat contacts in the 31–60 day window — outside the FCR measurement period |
| **DRL** — Distribution Replication Loss | 0.20 | Scenario mix drift over time (Jensen-Shannon divergence from baseline) — detects compositional gaming |
| **DOV** — Degradation of Validity | 0.25 | Month-over-month decay in proxy-truth agreement — the KPI is getting less accurate over time |
| **POR** — Proxy Overfit Ratio | 0.25 | Rate at which the proxy metric improves faster than the true outcome — the acceleration gap |

**SII ≥ 60** → **Veto.** Halt AI optimization. Re-audit the measurement environment before retraining any model that consumes these labels.

**SII 30–60** → **Watch.** Drift detected. Human review required before the next optimization cycle.

The SII doesn't replace your performance dashboard. It tells you whether your performance dashboard is still measuring performance.

---

## The Mechanism

This isn't a black-box score. The strongest mechanistic evidence in the dataset is the correlation between **rep gaming propensity** and the **proxy-true resolution gap**: r = +0.461, p < .001.

Contacts with higher gaming propensity produce a wider gap between what the system records and what's true. That gap compounds over months as gaming propensity drifts upward. The AI sees the proxy improving and optimizes harder against it. The proxy diverges further. The SII catches the divergence before it contaminates the next training cycle.

---

## Quick Start

```bash
pip install -r requirements.txt
python src/kpi_drift_observatory.py
```

Copy the 12 monthly `calls_sanitized_2025-*.csv` files from the NovaWireless Call Center Lab into `data/`.

---

## What It Produces

```
output/
├── evidence_report.json                SII computation with all four components
├── friction_decile_analysis.png        Proxy vs. true FCR across the friction gradient
├── logistic_regression_results.json    Full regression: resolution flag vs. churn
└── term_lift_analysis.json             Customer-side transcript term lift
```

---

## Repository Structure

```
NovaWireless_KPI_Drift_Observatory/
├── novawireless_paper.docx                 Primary paper: "When KPIs Lie"
├── novawireless_addendum.docx              Addendum A: Robustness & mechanism audits
├── src/
│   └── kpi_drift_observatory.py            Main analysis pipeline
├── data/
│   └── calls_sanitized_2025-*.csv          12 monthly CSVs from call generator
├── output/                                 Pipeline outputs (gitignored)
└── README.md
```

---

## Papers

> Aulabaugh, G. (2026). *When KPIs Lie: A System Integrity Framework for AI-Optimized Contact Center Operations.* NovaWireless KPI Drift Observatory — Technical Paper v1.0.

> Aulabaugh, G. (2026). *Robustness and Mechanism Audits for When KPIs Lie — Addendum A.* NovaWireless KPI Drift Observatory.

The addendum reports three robustness tests (lag prediction, within-rep slope, scenario holdout) and a mechanism audit establishing that TER neutrality is consistent with realistic churn architecture.

---

## Theoretical Foundation

The framework is grounded in three literatures that converge on the same warning:

**Goodhart's Law** (Goodhart, 1975; Strathern, 1997) — when a measure becomes a target, it ceases to be a good measure. The 42-point resolution gap is Goodhart's Law in CSV form.

**Campbell's Law** (Campbell, 1979) — quantitative indicators used for decision-making are subject to corruption pressures proportional to the stakes. Higher-stakes optimization produces faster proxy divergence.

**AI Alignment** (Amodei et al., 2016; Krakovna et al., 2020; Skalse et al., 2022) — specification gaming and reward hacking as formal failure modes. The call center is a live example: the AI optimizes against a reward signal that has decoupled from the outcome it was designed to track.

---

## Ecosystem Position

The Observatory sits at the top of the analytical stack. Where the governance pipeline scores individual calls and the transcript analysis identifies linguistic markers, the Observatory computes **system-level** integrity metrics that determine whether the measurement environment is still valid for AI optimization.

| Repository | Level | What It Does |
|---|---|---|
| novawireless-governance-pipeline | Call-level | Trust signal scoring, rep/scenario aggregation |
| novawireless-transcript-analysis | Call-level | Linguistic markers of proxy-outcome divergence |
| **KPI Drift Observatory** | **System-level** | **SII computation — the governance veto** |
| NovaFabric Validation Checklist | Signal-level | Causal validation of friction as upstream risk signal |

---

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `scipy`.

---

<p align="center">
  <b>Gina Aulabaugh</b><br>
  <a href="https://www.pixelkraze.com">www.pixelkraze.com</a>
</p>
