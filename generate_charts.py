"""
generate_charts.py
==================
Runs the complete analysis pipeline and saves all charts to outputs/charts/.
This script is the "headless" equivalent of running the notebook cells — useful
for CI/CD and for embedding images in README.md.

Run from the sales-force-effectiveness/ directory:
    python generate_charts.py
"""

import sys
import warnings
import os
from pathlib import Path

# ── make src importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import probplot

from src.modeling import (
    baseline_ols, fit_response_curve, fit_mixed_lm,
    compute_vif, residual_diagnostics
)
from src.optimization import (
    marginal_rx_curve, compute_segment_recommendations,
    optimal_call_frequency, DEFAULT_COST_PER_CALL, DEFAULT_REVENUE_PER_RX
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHART_DIR = Path("outputs/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)

PALETTE    = sns.color_palette("husl", 8)
SPEC_COLORS = {
    "GP":              "#4A90D9",
    "Cardiologist":    "#E05C5C",
    "Endocrinologist": "#50C878",
    "Neurologist":     "#F5A623",
}
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
})

def save(name: str):
    path = CHART_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading simulated data...")
df = pd.read_csv("outputs/panel_data.csv")
print(f"  {len(df)} rows | {df['doctor_id'].nunique()} doctors | "
      f"{df['territory_id'].nunique()} territories")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — EDA: Call Count Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating EDA charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(df["monthly_call_count"], bins=np.arange(0.5, 13.5, 1),
        color="#4A90D9", edgecolor="white", linewidth=0.8, alpha=0.85)
ax.set_xlabel("Monthly Call Count")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Monthly Calls per Doctor")
ax.set_xticks(range(1, 13))

ax = axes[1]
ax.hist(df["rx_volume"], bins=40, color="#E05C5C",
        edgecolor="white", linewidth=0.5, alpha=0.85)
ax.set_xlabel("Rx Volume (scripts/month)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Rx Volume")

fig.suptitle("EDA — Data Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save("01_eda_distributions")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — EDA: Calls vs Rx by Specialty (non-linearity motivation)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (spec, color) in zip(axes, SPEC_COLORS.items()):
    sub = df[df["doctor_specialty"] == spec]
    # Jitter calls slightly for visibility
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(sub))
    ax.scatter(sub["monthly_call_count"] + jitter, sub["rx_volume"],
               alpha=0.15, s=8, color=color)

    # Overlay mean Rx per call count
    means = sub.groupby("monthly_call_count")["rx_volume"].mean().reset_index()
    ax.plot(means["monthly_call_count"], means["rx_volume"],
            "o-", color=color, linewidth=2, markersize=7, label="Mean Rx")

    # Overlay MM curve with true params (ground truth reference)
    from data.simulate_data import SPECIALTIES, VMAX_BASE, KM
    vmax_true = VMAX_BASE * SPECIALTIES[spec]["vmax_mult"]
    calls_range = np.linspace(0.5, 12.5, 200)
    rx_true = (vmax_true * calls_range) / (KM + calls_range)
    ax.plot(calls_range, rx_true, "--", color="gray",
            linewidth=1.5, alpha=0.7, label="True MM curve")

    ax.set_title(f"{spec}", fontweight="bold")
    ax.set_xlabel("Monthly Calls")
    ax.set_ylabel("Rx Volume")
    ax.legend(fontsize=8)

fig.suptitle("Calls vs. Rx Volume by Specialty — Non-linearity Clearly Visible",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save("02_eda_calls_vs_rx_by_specialty")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — EDA: Territory-level Rx variability (motivates random effects)
# ═══════════════════════════════════════════════════════════════════════════
terr_means = (df.groupby("territory_id")["rx_volume"]
                .mean()
                .reset_index()
                .sort_values("rx_volume"))

fig, ax = plt.subplots(figsize=(13, 4))
bars = ax.bar(range(len(terr_means)), terr_means["rx_volume"],
              color="#4A90D9", alpha=0.85, edgecolor="white")
ax.axhline(terr_means["rx_volume"].mean(), color="#E05C5C",
           linewidth=1.8, linestyle="--", label="Overall mean")
ax.set_xticks(range(len(terr_means)))
ax.set_xticklabels(terr_means["territory_id"], rotation=45, ha="right")
ax.set_ylabel("Mean Rx Volume / Doctor-Month")
ax.set_title("Mean Rx Volume by Territory — Substantial Heterogeneity\n"
             "(Motivates Territory Random Effects in Mixed Model)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
save("03_eda_territory_heterogeneity")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — BASELINE MODEL: OLS fit + residual diagnostics
# ═══════════════════════════════════════════════════════════════════════════
print("\nFitting baseline OLS...")
ols_res = baseline_ols(df)

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)

# (a) Scatter + OLS line
ax1 = fig.add_subplot(gs[0])
jitter = np.random.default_rng(1).uniform(-0.3, 0.3, len(df))
ax1.scatter(df["monthly_call_count"] + jitter, df["rx_volume"],
            alpha=0.1, s=5, color="#4A90D9")
x_line = np.linspace(1, 12, 100)
coef   = ols_res["model"].params
ax1.plot(x_line, coef["Intercept"] + coef["monthly_call_count"] * x_line,
         "r-", linewidth=2.5, label=f"OLS (R²={ols_res['rsq']:.3f})")
ax1.set_xlabel("Monthly Calls")
ax1.set_ylabel("Rx Volume")
ax1.set_title("OLS Fit\n(Misses saturation)", fontweight="bold")
ax1.legend()

# (b) Residuals vs Fitted
ax2 = fig.add_subplot(gs[1])
ax2.scatter(ols_res["predictions"], ols_res["residuals"],
            alpha=0.15, s=5, color="#E05C5C")
ax2.axhline(0, color="black", linewidth=1)
ax2.set_xlabel("Fitted Values")
ax2.set_ylabel("Residuals")
ax2.set_title("Residuals vs Fitted\n(Curvature & heteroscedasticity)", fontweight="bold")

# (c) Q-Q plot
ax3 = fig.add_subplot(gs[2])
(osm, osr), (slope, intercept, r) = probplot(ols_res["residuals"], dist="norm")
ax3.scatter(osm, osr, alpha=0.3, s=5, color="#50C878")
ax3.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=2)
ax3.set_xlabel("Theoretical Quantiles")
ax3.set_ylabel("Sample Quantiles")
ax3.set_title("Q-Q Plot\n(Residuals non-normal)", fontweight="bold")

fig.suptitle(f"Baseline OLS Diagnostics  |  AIC={ols_res['aic']:.0f}  BIC={ols_res['bic']:.0f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save("04_baseline_ols_diagnostics")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — RESPONSE CURVE: MM fit per specialty
# ═══════════════════════════════════════════════════════════════════════════
print("\nFitting response curves per specialty...")
rc_res = fit_response_curve(df, group_col="doctor_specialty")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
calls_range = np.linspace(0.1, 12, 200)

for ax, (spec, color) in zip(axes, SPEC_COLORS.items()):
    sub    = df[df["doctor_specialty"] == spec]
    params = rc_res["params"][spec]

    if not params.get("converged", True):
        ax.set_title(f"{spec} — DID NOT CONVERGE")
        continue

    vmax, km = params["vmax"], params["km"]
    fitted_curve = (vmax * calls_range) / (km + calls_range)

    # Scatter
    jitter = np.random.default_rng(2).uniform(-0.3, 0.3, len(sub))
    ax.scatter(sub["monthly_call_count"] + jitter, sub["rx_volume"],
               alpha=0.12, s=8, color=color)

    # Fitted MM curve
    ax.plot(calls_range, fitted_curve, "-", color=color,
            linewidth=2.5, label=f"MM fit: Vmax={vmax:.1f}, Km={km:.2f}")

    # Mark optimal call frequency
    opt = optimal_call_frequency(vmax, km)
    n_star = opt["optimal_calls"]
    rx_star = opt["rx_at_optimal"]
    ax.axvline(n_star, color="black", linewidth=1.5, linestyle="--")
    ax.scatter([n_star], [rx_star], color="black", s=100, zorder=5,
               label=f"Optimal = {n_star} calls")

    ax.set_title(f"{spec}", fontweight="bold")
    ax.set_xlabel("Monthly Calls")
    ax.set_ylabel("Rx Volume")
    ax.legend(fontsize=8.5)

fig.suptitle(f"Michaelis-Menten Response Curves by Specialty\n"
             f"Total AIC={rc_res['aic']:.0f}  (vs OLS AIC={ols_res['aic']:.0f})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save("05_response_curves_by_specialty")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 6 — MIXED EFFECTS: diagnostics
# ═══════════════════════════════════════════════════════════════════════════
print("\nFitting mixed-effects model...")
me_res = fit_mixed_lm(df)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Residuals vs fitted (log scale)
ax = axes[0]
ax.scatter(me_res["predictions"], me_res["residuals"],
           alpha=0.15, s=5, color="#4A90D9")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Fitted log(Rx)")
ax.set_ylabel("Residuals")
ax.set_title("MixedLM Residuals vs Fitted", fontweight="bold")

# (b) Q-Q of residuals
ax = axes[1]
(osm, osr), (slope, intercept, r) = probplot(me_res["residuals"], dist="norm")
ax.scatter(osm, osr, alpha=0.3, s=5, color="#E05C5C")
ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=2)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Q-Q Plot (MixedLM residuals)", fontweight="bold")

# (c) Random effects distribution (territory intercepts)
ax = axes[2]
re_vals = [v.iloc[0] for v in me_res["random_effects"].values()]
ax.hist(re_vals, bins=15, color="#50C878", edgecolor="white", alpha=0.9)
ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Random Effect (territory intercept)")
ax.set_ylabel("Count")
ax.set_title("Territory Random Effects Distribution\n(Captured, not pooled away)",
             fontweight="bold")

fig.suptitle(f"Mixed-Effects Model Diagnostics  |  AIC={me_res['aic']:.0f}  "
             f"BIC={me_res['bic']:.0f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save("06_mixedlm_diagnostics")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 7 — MODEL COMPARISON: AIC/BIC bar chart
# ═══════════════════════════════════════════════════════════════════════════
models    = ["Baseline OLS", "MM Curve\n(per specialty)", "Mixed LM\n(log-linear)"]
aic_vals  = [ols_res["aic"], rc_res["aic"], me_res["aic"]]
bic_vals  = [ols_res["bic"], rc_res["bic"], me_res["bic"]]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, aic_vals, width, label="AIC", color="#4A90D9", alpha=0.85)
bars2 = ax.bar(x + width/2, bic_vals, width, label="BIC", color="#E05C5C", alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Information Criterion (lower = better)")
ax.set_title("Model Comparison: AIC & BIC\n"
             "(Lower values indicate better fit-to-complexity tradeoff)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
save("07_model_comparison_aic_bic")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 8 — OPTIMIZATION: Marginal Rx & Revenue vs calls (all specialties)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating optimization charts...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (spec, color) in zip(axes, SPEC_COLORS.items()):
    params = rc_res["params"][spec]
    if not params.get("converged", True):
        continue
    vmax, km = params["vmax"], params["km"]

    curve = marginal_rx_curve(vmax, km)
    curve["marginal_rev"] = curve["marginal_rx"] * DEFAULT_REVENUE_PER_RX

    opt    = optimal_call_frequency(vmax, km)
    n_star = opt["optimal_calls"]

    c = curve[curve["calls"] > 0]
    ax.bar(c["calls"], c["marginal_rev"], color=color, alpha=0.75, label="Marginal revenue ($)")
    ax.axhline(DEFAULT_COST_PER_CALL, color="black", linewidth=1.8,
               linestyle="--", label=f"Cost/call (${DEFAULT_COST_PER_CALL:.0f})")
    ax.axvline(n_star, color="red", linewidth=2, linestyle="-",
               label=f"n* = {n_star} calls")

    ax.set_xlabel("Monthly Calls")
    ax.set_ylabel("Marginal Revenue ($)")
    ax.set_title(f"{spec}", fontweight="bold")
    ax.legend(fontsize=8)

fig.suptitle("Marginal Revenue per Additional Call vs. Cost per Call\n"
             "Optimal call frequency n* = last profitable call",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save("08_marginal_revenue_optimization")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 9 — RECOMMENDATION TABLE: visual summary
# ═══════════════════════════════════════════════════════════════════════════
spec_params = {
    spec: rc_res["params"][spec]
    for spec in rc_res["params"]
    if rc_res["params"][spec].get("converged", True)
}
reco_df = compute_segment_recommendations(
    spec_params, df,
    segment_col="doctor_specialty",
    cost_per_call=DEFAULT_COST_PER_CALL,
    revenue_per_rx=DEFAULT_REVENUE_PER_RX,
)
print("\nSegment Recommendation Table:")
print(reco_df.to_string(index=False))

# Save to CSV
reco_df.to_csv("outputs/segment_recommendations.csv", index=False)

# Visual: bar chart of current vs optimal calls
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
x = np.arange(len(reco_df))
w = 0.35
b1 = ax.bar(x - w/2, reco_df["Current Calls/Month"], w,
            label="Current", color="#4A90D9", alpha=0.85)
b2 = ax.bar(x + w/2, reco_df["Optimal Calls/Month"], w,
            label="Optimal (n*)", color="#E05C5C", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(reco_df["Segment"], rotation=20, ha="right")
ax.set_ylabel("Calls / Month")
ax.set_title("Current vs. Optimal Call Frequency by Segment", fontweight="bold")
ax.legend()

ax = axes[1]
colors = ["#50C878" if v >= 0 else "#E05C5C"
          for v in reco_df["Net Monthly Impact/Doc ($)"]]
ax.bar(reco_df["Segment"], reco_df["Net Monthly Impact/Doc ($)"],
       color=colors, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Net Monthly Impact / Doctor ($)")
ax.set_title("Projected Net Impact per Doctor/Month\n"
             "(Revenue uplift – call cost change)", fontweight="bold")

fig.suptitle("Segment-Level Call Frequency Recommendations",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save("09_segment_recommendations")


# ═══════════════════════════════════════════════════════════════════════════
# CHART 10 — RESPONSE CURVES WITH OPERATING POINT (for README hero chart)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
calls_range = np.linspace(0.1, 12, 200)

for spec, color in SPEC_COLORS.items():
    params = rc_res["params"].get(spec, {})
    if not params.get("converged", True):
        continue
    vmax, km = params["vmax"], params["km"]
    fitted_curve = (vmax * calls_range) / (km + calls_range)
    ax.plot(calls_range, fitted_curve, "-", color=color, linewidth=2.5, label=spec)

    opt    = optimal_call_frequency(vmax, km)
    n_star = opt["optimal_calls"]
    rx_star = opt["rx_at_optimal"]
    ax.scatter([n_star], [rx_star], color=color, s=150, zorder=6,
               edgecolors="black", linewidth=1.5)
    ax.annotate(f"n*={n_star}", (n_star, rx_star),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, color=color, fontweight="bold")

ax.set_xlabel("Monthly Calls per Doctor", fontsize=12)
ax.set_ylabel("Expected Rx Volume (scripts/month)", fontsize=12)
ax.set_title("Michaelis-Menten Response Curves by Specialty\n"
             "● = ROI-maximising call frequency (n*)",
             fontsize=14, fontweight="bold")
ax.legend(title="Specialty", fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save("10_response_curves_hero")

print("\nDone! All charts saved to outputs/charts/")
print(f"\nFinal model AIC comparison:")
print(f"  Baseline OLS : {ols_res['aic']:.1f}")
print(f"  MM Curve     : {rc_res['aic']:.1f}  (Delta={ols_res['aic']-rc_res['aic']:.1f} better)")
print(f"  Mixed LM     : {me_res['aic']:.1f}")
