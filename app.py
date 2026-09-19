"""
app.py — Streamlit Dashboard: Pharma Sales Force Effectiveness
==============================================================

A lightweight, interactive dashboard for exploring the fitted response curves,
optimal call frequency recommendations, and cost sensitivity analysis per
doctor segment / territory.

RUN:
    streamlit run app.py

No authentication, no database. All model fitting runs on startup (~3s),
then the UI is interactive.

ARCHITECTURE:
- Uses @st.cache_data to fit response curves only once per session.
- All interactive widgets (dropdown, slider) trigger re-computation of
  the optimization logic (fast: pure arithmetic on fitted parameters).
- Charts rendered via matplotlib and displayed inline with st.pyplot().
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.modeling import fit_response_curve
from src.optimization import (
    marginal_rx_curve,
    optimal_call_frequency,
    compute_segment_recommendations,
    compute_territory_recommendations,
    sensitivity_table,
    DEFAULT_COST_PER_CALL,
    DEFAULT_REVENUE_PER_RX,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma SFE — Response Curve Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* Cards / metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2433 0%, #252d3f 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric label */
    div[data-testid="metric-container"] label {
        color: #8b9ab5 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Metric value */
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #e8eaf6 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Metric delta */
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 0.85rem !important;
    }
    
    /* Section headers */
    h1 { 
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }
    h2 { color: #93c5fd !important; font-size: 1.3rem !important; }
    h3 { color: #c4b5fd !important; font-size: 1.1rem !important; }
    
    /* DataFrames */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    
    /* Info / warning boxes */
    .stAlert { border-radius: 10px; }
    
    /* Slider */
    .stSlider > div { padding-top: 0.5rem; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] { 
        font-weight: 600; 
        color: #8b9ab5;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e2433, #252d3f) !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #60a5fa !important;
    }
    
    /* Selectbox */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Divider */
    hr { border-color: #2d3748; margin: 1.5rem 0; }
    
    /* Caption / footnote text */
    .footnote { 
        color: #64748b; 
        font-size: 0.78rem; 
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1e2433"
TEXT_COL  = "#e8eaf6"
GRID_COL  = "#2d3748"
ACCENT    = "#60a5fa"
RED       = "#f87171"
GREEN     = "#34d399"
YELLOW    = "#fbbf24"
PURPLE    = "#a78bfa"

SPEC_COLORS = {
    "GP":              "#60a5fa",
    "Cardiologist":    "#f87171",
    "Endocrinologist": "#34d399",
    "Neurologist":     "#fbbf24",
}

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "text.color":        TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COL,
    "legend.labelcolor": TEXT_COL,
})


# ─────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and fitting models…")
def load_and_fit():
    data_path = Path(__file__).parent / "outputs" / "panel_data.csv"
    if not data_path.exists():
        # Fallback: re-run simulation
        import subprocess, sys
        subprocess.run(
            [sys.executable, "data/simulate_data.py"],
            cwd=str(Path(__file__).parent), check=True
        )
    df = pd.read_csv(data_path)

    # Fit MM curves per specialty
    spec_rc  = fit_response_curve(df, group_col="doctor_specialty")
    # Fit MM curves per territory
    terr_rc  = fit_response_curve(df, group_col="territory_id")

    return df, spec_rc, terr_rc

df, spec_rc, terr_rc = load_and_fit()


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 SFE Dashboard")
    st.markdown("*Sales Force Effectiveness*\n*& Call Response Modeling*")
    st.divider()

    st.markdown("### 🎯 Analysis Scope")
    scope = st.radio("Group by:", ["Doctor Specialty", "Territory"], index=0)

    if scope == "Doctor Specialty":
        entities      = sorted(df["doctor_specialty"].unique())
        rc_params_all = spec_rc["params"]
        seg_col       = "doctor_specialty"
    else:
        entities      = sorted(df["territory_id"].unique())
        rc_params_all = terr_rc["params"]
        seg_col       = "territory_id"

    selected = st.selectbox("Select segment:", entities)

    st.divider()
    st.markdown("### 💰 Economic Assumptions")

    st.caption("Adjust cost & revenue levers to see how the optimal call frequency changes.")

    cost_per_call = st.slider(
        "Cost per call (USD)",
        min_value=50, max_value=600, value=250, step=25,
        help="Includes rep salary, travel, samples, overhead. Industry range: $200–$350."
    )
    rev_per_rx = st.slider(
        "Net revenue per Rx (USD)",
        min_value=50, max_value=400, value=150, step=10,
        help="Net realised price per additional script after payer rebates."
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "**Pharma SFE Response Curve Modeling** · "
        "Built with statsmodels, scipy.optimize, and Streamlit. "
        "Simulated data only — not real patient or prescriber data."
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("# 💊 Pharma Sales Force Effectiveness")
st.markdown("### Response Curve Modeling & Call Frequency Optimization")
st.markdown(
    "This dashboard explores the **diminishing-returns relationship** between "
    "sales rep call frequency and physician Rx volume, identifies the "
    "**ROI-maximising call frequency** per segment, and projects the business "
    "impact of rebalancing call plans."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Response Curve",
    "📊 Segment Overview",
    "🔧 Sensitivity Analysis",
    "📋 Recommendation Table",
])


# ─── TAB 1: RESPONSE CURVE ───────────────────
with tab1:
    seg_params = rc_params_all.get(selected, {})

    if not seg_params or not seg_params.get("converged", True):
        st.warning(f"Model did not converge for {selected}. Try another segment.")
    else:
        vmax = seg_params["vmax"]
        km   = seg_params["km"]

        # Compute optimal with current slider values
        opt    = optimal_call_frequency(vmax, km, cost_per_call, rev_per_rx)
        n_star = opt["optimal_calls"]
        rx_star = opt["rx_at_optimal"]

        # Observed data for this segment
        seg_df = df[df[seg_col] == selected]
        current_calls = seg_df["monthly_call_count"].mean()
        current_rx    = seg_df["rx_volume"].mean()

        # ── METRIC ROW
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fitted Vmax (Rx ceiling)",
                    f"{vmax:.1f}", help="Asymptotic max Rx at infinite calls")
        col2.metric("Half-saturation Km",
                    f"{km:.2f} calls", help="Calls at which Rx reaches Vmax/2")
        col3.metric("Optimal Call Freq (n*)",
                    f"{n_star} calls/mo",
                    delta=f"{n_star - round(current_calls):.0f} vs current")
        col4.metric("Projected Rx at n*",
                    f"{rx_star:.1f}",
                    delta=f"{rx_star - current_rx:+.1f} vs current avg")

        st.markdown("---")

        # ── RESPONSE CURVE CHART
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        calls_range = np.linspace(0.1, 12, 300)

        color = SPEC_COLORS.get(selected, ACCENT)

        # (a) Response curve
        ax = axes[0]
        fitted_curve = (vmax * calls_range) / (km + calls_range)
        ax.plot(calls_range, fitted_curve, "-", color=color,
                linewidth=2.5, label="Fitted MM curve")

        # Shade diminishing-returns zone (n* onwards)
        ax.fill_between(calls_range, fitted_curve, 0,
                        where=(calls_range <= n_star),
                        alpha=0.12, color=GREEN, label="Economically viable zone")
        ax.fill_between(calls_range, fitted_curve, 0,
                        where=(calls_range > n_star),
                        alpha=0.08, color=RED, label="Sub-optimal zone (over-call)")

        # Current operating point
        rx_at_current = (vmax * current_calls) / (km + current_calls)
        ax.scatter([current_calls], [rx_at_current],
                   color=YELLOW, s=180, zorder=7, edgecolors="white",
                   linewidth=1.5, label=f"Current: {current_calls:.1f} calls")

        # Optimal operating point
        ax.scatter([n_star], [rx_star],
                   color=GREEN, s=200, zorder=8, marker="*",
                   edgecolors="white", linewidth=1.5,
                   label=f"Optimal n*: {n_star} calls")

        ax.axvline(n_star, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("Monthly Calls per Doctor")
        ax.set_ylabel("Expected Rx Volume")
        ax.set_title(f"{selected} — Response Curve", fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(True)

        # (b) Marginal revenue vs cost
        ax = axes[1]
        curve = marginal_rx_curve(vmax, km)
        c     = curve[curve["calls"] > 0].copy()
        c["marginal_rev"] = c["marginal_rx"] * rev_per_rx

        bar_colors = [GREEN if row["calls"] <= n_star else RED
                      for _, row in c.iterrows()]
        ax.bar(c["calls"], c["marginal_rev"], color=bar_colors, alpha=0.85,
               edgecolor=DARK_BG, linewidth=0.8)
        ax.axhline(cost_per_call, color=YELLOW, linewidth=2, linestyle="--",
                   label=f"Cost/call = ${cost_per_call}")
        ax.axvline(n_star + 0.5, color=GREEN, linewidth=1.5, linestyle="--",
                   alpha=0.6, label=f"n* = {n_star}")

        ax.set_xlabel("Monthly Calls")
        ax.set_ylabel("Marginal Revenue per Additional Call ($)")
        ax.set_title("Marginal Revenue vs Cost per Call", fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(True)

        plt.tight_layout(pad=2.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            f'<p class="footnote">💡 <b>Reading this chart:</b> Green bars = calls where '
            f'marginal revenue > cost (profitable). Red bars = calls that cost more than '
            f'they generate. The yellow dotted line is your cost-per-call assumption '
            f'(${cost_per_call}). Adjust the slider in the sidebar to see how this changes.</p>',
            unsafe_allow_html=True,
        )


# ─── TAB 2: SEGMENT OVERVIEW ─────────────────
with tab2:
    st.markdown("### All Segments — Response Curves")
    st.caption(
        "Each panel shows the fitted Michaelis-Menten curve for a specialty, "
        "with the current operating point (●) and optimal n* (★) marked."
    )

    if scope == "Doctor Specialty":
        panels = list(SPEC_COLORS.keys())
        panel_colors = SPEC_COLORS
    else:
        panels = sorted(df["territory_id"].unique())[:8]  # show first 8 territories
        panel_colors = {t: ACCENT for t in panels}

    n_panels = len(panels)
    ncols    = min(4, n_panels)
    nrows    = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    calls_range = np.linspace(0.1, 12, 300)

    for ax, panel in zip(axes, panels):
        params = rc_params_all.get(panel, {})
        color  = panel_colors.get(panel, ACCENT)

        if not params or not params.get("converged", True):
            ax.set_title(f"{panel}\n(No convergence)", color=TEXT_COL)
            continue

        vmax, km = params["vmax"], params["km"]
        fitted_curve = (vmax * calls_range) / (km + calls_range)

        seg_sub  = df[df[seg_col] == panel]
        cur_calls = seg_sub["monthly_call_count"].mean()
        cur_rx    = seg_sub["rx_volume"].mean()

        opt     = optimal_call_frequency(vmax, km, cost_per_call, rev_per_rx)
        n_star  = opt["optimal_calls"]
        rx_star = opt["rx_at_optimal"]

        ax.plot(calls_range, fitted_curve, "-", color=color, linewidth=2.2)
        cur_rx_curve = (vmax * cur_calls) / (km + cur_calls)
        ax.scatter([cur_calls], [cur_rx_curve], color=YELLOW, s=120,
                   zorder=7, edgecolors="white", linewidth=1.2)
        ax.scatter([n_star], [rx_star], color=GREEN, s=150, zorder=8,
                   marker="*", edgecolors="white", linewidth=1.2)
        ax.axvline(n_star, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)

        ax.set_title(
            f"{panel}\nVmax={vmax:.0f} | Km={km:.1f} | n*={n_star}",
            fontweight="bold", fontsize=10
        )
        ax.set_xlabel("Calls", fontsize=9)
        ax.set_ylabel("Rx", fontsize=9)
        ax.grid(True)

    # Hide unused axes
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ─── TAB 3: SENSITIVITY ANALYSIS ─────────────
with tab3:
    st.markdown("### Cost Sensitivity: How does n* change as cost-per-call varies?")
    st.caption(
        "The optimal call frequency is sensitive to the cost-per-call assumption. "
        "This analysis shows how n* shifts across the full cost range, holding "
        f"revenue per Rx constant at ${rev_per_rx}."
    )

    seg_params_sens = rc_params_all.get(selected, {})
    if seg_params_sens and seg_params_sens.get("converged", True):
        sens_df = sensitivity_table(
            seg_params_sens["vmax"], seg_params_sens["km"],
            cost_range=(50, 600),
            revenue_per_rx=rev_per_rx,
            steps=12,
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.step(sens_df["cost_per_call ($)"], sens_df["optimal_calls"],
                where="post", color=ACCENT, linewidth=2.5)
        ax.scatter(sens_df["cost_per_call ($)"], sens_df["optimal_calls"],
                   color=ACCENT, s=60, zorder=5)
        ax.axvline(cost_per_call, color=YELLOW, linewidth=2, linestyle="--",
                   label=f"Current assumption: ${cost_per_call}")
        ax.set_xlabel("Cost per Call (USD)")
        ax.set_ylabel("Optimal Call Frequency n*")
        ax.set_title(f"{selected} — n* vs Cost per Call", fontweight="bold")
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        ax.plot(sens_df["cost_per_call ($)"], sens_df["roi_at_optimal (%)"],
                color=PURPLE, linewidth=2.5, marker="o", markersize=6)
        ax.axhline(0, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axvline(cost_per_call, color=YELLOW, linewidth=2, linestyle="--",
                   label=f"Current: ${cost_per_call}")
        ax.set_xlabel("Cost per Call (USD)")
        ax.set_ylabel("ROI at Optimal Frequency (%)")
        ax.set_title(f"{selected} — ROI vs Cost per Call", fontweight="bold")
        ax.legend()
        ax.grid(True)

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.dataframe(sens_df.style.format({
            "cost_per_call ($)": "${:.0f}",
            "rx_at_optimal": "{:.1f}",
            "roi_at_optimal (%)": "{:.1f}%",
        }).background_gradient(subset=["roi_at_optimal (%)"],
                               cmap="RdYlGn", vmin=-50, vmax=200),
            use_container_width=True, hide_index=True)
    else:
        st.warning("No converged model parameters for selected segment.")


# ─── TAB 4: RECOMMENDATION TABLE ─────────────
with tab4:
    st.markdown("### Segment-Level Call Frequency Recommendations")
    st.caption(
        f"Based on: cost per call = **${cost_per_call}** | "
        f"revenue per Rx = **${rev_per_rx}** | "
        f"Grouped by: **{scope}**"
    )

    # Rebuild recommendations with current slider values
    converged_params = {
        seg: p for seg, p in rc_params_all.items()
        if p.get("converged", True) and "vmax" in p
    }
    reco = compute_segment_recommendations(
        converged_params, df,
        segment_col=seg_col,
        cost_per_call=cost_per_call,
        revenue_per_rx=rev_per_rx,
    )

    # Color-code the Action column
    def style_action(val):
        if val == "Increase":
            return "color: #34d399; font-weight: bold"
        elif val == "Decrease":
            return "color: #f87171; font-weight: bold"
        return "color: #fbbf24; font-weight: bold"

    styled = reco.style.map(style_action, subset=["Action"]) \
                       .format({
                           "Fitted Vmax": "{:.1f}",
                           "Fitted Km": "{:.2f}",
                           "Current Calls/Month": "{:.1f}",
                           "Projected DeltaRx/Doc/Month": "{:+.1f}",
                           "Projected DeltaRev/Doc/Month ($)": "${:+.0f}",
                           "Net Monthly Impact/Doc ($)": "${:+.0f}",
                       }) \
                       .background_gradient(
                           subset=["Net Monthly Impact/Doc ($)"],
                           cmap="RdYlGn", vmin=-500, vmax=500
                       )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📌 Key Takeaways")

    for _, row in reco.iterrows():
        action_emoji = "🔼" if row["Action"] == "Increase" else "🔽" if row["Action"] == "Decrease" else "✅"
        impact_str = f"${row['Net Monthly Impact/Doc ($)']:+.0f}/doctor/month"
        st.markdown(
            f"- **{row['Segment']}** ({action_emoji} {row['Action']}): "
            f"Move from **{row['Current Calls/Month']:.1f}->{row['Optimal Calls/Month']}** "
            f"calls/month — projected net impact: **{impact_str}**"
        )

    st.markdown("---")
    st.markdown(
        '<p class="footnote">⚠️ <b>Limitations:</b> Recommendations are based on '
        "simulated data and stated assumptions. In production, calibrate cost-per-call "
        "and revenue-per-Rx with actual finance and market data. Mixed-effects territory "
        "random effects are estimated but not incorporated here — see modeling.py for "
        "the full MixedLM results.</p>",
        unsafe_allow_html=True,
    )

    # Download button
    csv = reco.to_csv(index=False)
    st.download_button(
        "⬇️ Download Recommendation Table (CSV)",
        data=csv,
        file_name=f"sfe_recommendations_{scope.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
