"""
optimization.py — Marginal ROI Analysis & Call-Frequency Recommendation Engine
================================================================================

BUSINESS LOGIC
--------------
Given a fitted Michaelis-Menten response curve per doctor segment, this module:

1. Computes the MARGINAL Rx gain from the n-th call:
       ΔRx(n) = Rx(n) - Rx(n-1)
              = Vmax * [n/(Km+n) - (n-1)/(Km+n-1)]

2. Converts marginal Rx to marginal REVENUE:
       ΔREV(n) = ΔRx(n) × revenue_per_rx

   ASSUMPTION: revenue_per_rx = $150 USD per additional Rx written.
   Basis: mid-range branded drug net realised price after rebates/payer mix,
   representative of specialty pharma categories (e.g., diabetes, cardiology).
   This is a *stated assumption* — sensitivity analysis is available via the
   Streamlit slider.

3. Compares marginal revenue to COST PER CALL:
       ASSUMPTION: cost_per_call = $250 USD.
       Basis: ZS Associates / industry benchmarks cite $200–$350 per
       physician call when including rep salary, benefits, travel, samples,
       and overhead (Stelmach, 2019; IQVIA estimates). $250 is the midpoint.

4. Finds the OPTIMAL call frequency n* where:
       ΔREV(n*) ≥ cost_per_call  AND  ΔREV(n*+1) < cost_per_call
   i.e., the last call that "pays for itself".

5. Produces a SEGMENT RECOMMENDATION TABLE comparing current average
   call frequency (from observed data) vs. the optimum, with projected
   monthly Rx impact and revenue uplift per doctor.

USAGE
-----
All functions are pure (no side effects). The Streamlit app calls
compute_segment_recommendations() after updating cost_per_call from the slider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (explicit assumptions — all overridable)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COST_PER_CALL  = 250.0   # USD: midpoint of industry benchmark range
DEFAULT_REVENUE_PER_RX = 150.0   # USD: net revenue per incremental Rx written
MAX_CALLS_EVALUATED    = 12      # upper bound for call frequency analysis


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARGINAL RX GAIN CURVE
# ─────────────────────────────────────────────────────────────────────────────

def marginal_rx_curve(vmax: float,
                      km: float,
                      max_calls: int = MAX_CALLS_EVALUATED) -> pd.DataFrame:
    """
    Computes Rx volume and marginal Rx gain for call frequencies 0..max_calls.

    Formula:
        Rx(n) = (Vmax * n) / (Km + n)
        ΔRx(n) = Rx(n) - Rx(n-1)     [first call gain relative to 0 baseline]

    Parameters
    ----------
    vmax      : fitted Vmax parameter for this segment
    km        : fitted Km parameter for this segment
    max_calls : upper bound for call frequency range

    Returns
    -------
    DataFrame with columns:
        calls, rx_volume, marginal_rx
    """
    calls = np.arange(0, max_calls + 1, dtype=float)
    rx = (vmax * calls) / (km + calls)
    rx[0] = 0.0  # explicit: 0 calls → 0 Rx from detailing

    marginal_rx = np.diff(rx, prepend=0.0)
    marginal_rx[0] = 0.0  # no gain at 0 calls

    return pd.DataFrame({
        "calls":       calls.astype(int),
        "rx_volume":   rx,
        "marginal_rx": marginal_rx,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIMAL CALL FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def optimal_call_frequency(vmax: float,
                           km: float,
                           cost_per_call: float = DEFAULT_COST_PER_CALL,
                           revenue_per_rx: float = DEFAULT_REVENUE_PER_RX,
                           max_calls: int = MAX_CALLS_EVALUATED) -> dict[str, Any]:
    """
    Finds the ROI-maximising call frequency for a segment with given (Vmax, Km).

    The optimum n* is the highest call count where:
        marginal_rx(n) × revenue_per_rx ≥ cost_per_call

    If this condition is never met (even the first call doesn't pay off),
    n* = 0 (no calls recommended). If it holds all the way to max_calls,
    n* = max_calls (may indicate an under-serviced segment).

    Parameters
    ----------
    vmax, km       : MM curve parameters
    cost_per_call  : USD cost per sales rep visit
    revenue_per_rx : USD net revenue per Rx written
    max_calls      : upper evaluation limit

    Returns
    -------
    dict with:
        'optimal_calls'      : int, n*
        'marginal_curve'     : DataFrame from marginal_rx_curve()
        'break_even_rx'      : float, ΔRx needed to justify one call
        'rx_at_optimal'      : float, total Rx at n*
        'roi_at_optimal'     : float, (rev - cost) / cost at n*
    """
    curve = marginal_rx_curve(vmax, km, max_calls)

    # Break-even ΔRx per call
    break_even_rx = cost_per_call / revenue_per_rx

    # Find highest call where marginal Rx × revenue ≥ cost
    profitable = curve[curve["calls"] > 0].copy()
    profitable["marginal_rev"] = profitable["marginal_rx"] * revenue_per_rx
    profitable["profitable"]   = profitable["marginal_rev"] >= cost_per_call

    if profitable["profitable"].any():
        n_star = int(profitable[profitable["profitable"]]["calls"].max())
    else:
        n_star = 0

    rx_at_opt = float(curve.loc[curve["calls"] == n_star, "rx_volume"].iloc[0])

    # Cumulative ROI at n*: total_rev - total_cost
    total_rev  = rx_at_opt * revenue_per_rx
    total_cost = n_star * cost_per_call
    roi = (total_rev - total_cost) / max(total_cost, 1e-6)

    return {
        "optimal_calls":   n_star,
        "marginal_curve":  curve,
        "break_even_rx":   break_even_rx,
        "rx_at_optimal":   rx_at_opt,
        "roi_at_optimal":  roi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEGMENT RECOMMENDATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_segment_recommendations(segment_params: dict[str, dict],
                                    observed_df: pd.DataFrame,
                                    segment_col: str = "doctor_specialty",
                                    call_col: str = "monthly_call_count",
                                    rx_col: str = "rx_volume",
                                    cost_per_call: float = DEFAULT_COST_PER_CALL,
                                    revenue_per_rx: float = DEFAULT_REVENUE_PER_RX
                                    ) -> pd.DataFrame:
    """
    Builds the segment-level recommendation table.

    For each segment (specialty or territory):
    - Reads fitted Vmax, Km from segment_params
    - Computes optimal call frequency
    - Compares to current observed mean call frequency
    - Estimates monthly Rx and revenue impact of moving to optimal

    Parameters
    ----------
    segment_params : dict mapping segment label → {'vmax': float, 'km': float}
    observed_df    : raw panel data (for computing observed call frequencies)
    segment_col    : column identifying the segment
    call_col       : call count column
    rx_col         : Rx volume column
    cost_per_call  : USD cost per call assumption
    revenue_per_rx : USD net revenue per Rx

    Returns
    -------
    pd.DataFrame — recommendation table
    """
    records = []

    for seg, params in segment_params.items():
        if not params.get("converged", True):
            continue

        vmax = params["vmax"]
        km   = params["km"]

        # Observed data for this segment
        seg_df = observed_df[observed_df[segment_col] == seg]
        current_calls = float(seg_df[call_col].mean())
        current_rx    = float(seg_df[rx_col].mean())

        # Optimal
        opt = optimal_call_frequency(vmax, km, cost_per_call, revenue_per_rx)
        n_star  = opt["optimal_calls"]
        rx_star = opt["rx_at_optimal"]

        # Rx and revenue impact per doctor per month
        rx_delta  = rx_star - current_rx
        rev_delta = rx_delta * revenue_per_rx

        # Call cost impact
        call_delta     = n_star - current_calls
        call_cost_delta = call_delta * cost_per_call

        # Net monthly impact per doctor
        net_impact = rev_delta - call_cost_delta

        records.append({
            "Segment":                seg,
            "Fitted Vmax":            round(vmax, 1),
            "Fitted Km":              round(km, 2),
            "Current Calls/Month":    round(current_calls, 1),
            "Optimal Calls/Month":    n_star,
            "Call Change":            round(n_star - current_calls, 1),
            "Action":                 ("Increase" if n_star > current_calls
                                       else "Decrease" if n_star < current_calls
                                       else "Maintain"),
            "Projected DeltaRx/Doc/Month":  round(rx_delta, 1),
            "Projected DeltaRev/Doc/Month ($)": round(rev_delta, 1),
            "Net Monthly Impact/Doc ($)":   round(net_impact, 1),
        })

    return pd.DataFrame(records).sort_values("Net Monthly Impact/Doc ($)",
                                             ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TERRITORY-LEVEL RECOMMENDATIONS (same logic, different grouping)
# ─────────────────────────────────────────────────────────────────────────────

def compute_territory_recommendations(territory_params: dict[str, dict],
                                      observed_df: pd.DataFrame,
                                      cost_per_call: float = DEFAULT_COST_PER_CALL,
                                      revenue_per_rx: float = DEFAULT_REVENUE_PER_RX
                                      ) -> pd.DataFrame:
    """
    Same as compute_segment_recommendations but keyed by territory.
    Useful for the Streamlit territory selector dropdown.
    """
    return compute_segment_recommendations(
        segment_params=territory_params,
        observed_df=observed_df,
        segment_col="territory_id",
        call_col="monthly_call_count",
        rx_col="rx_volume",
        cost_per_call=cost_per_call,
        revenue_per_rx=revenue_per_rx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. SENSITIVITY ANALYSIS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_table(vmax: float,
                      km: float,
                      cost_range: tuple[float, float] = (100, 500),
                      revenue_per_rx: float = DEFAULT_REVENUE_PER_RX,
                      steps: int = 9) -> pd.DataFrame:
    """
    Returns a table of optimal call frequencies across a range of cost-per-call
    assumptions. Used in the Streamlit sensitivity tab.

    Parameters
    ----------
    vmax, km     : MM parameters
    cost_range   : (min_cost, max_cost) in USD
    revenue_per_rx: USD per Rx
    steps        : number of cost values to evaluate

    Returns
    -------
    DataFrame with columns: cost_per_call, optimal_calls, rx_at_optimal, roi
    """
    costs = np.linspace(cost_range[0], cost_range[1], steps)
    rows = []
    for c in costs:
        res = optimal_call_frequency(vmax, km, c, revenue_per_rx)
        rows.append({
            "cost_per_call ($)":    round(c, 0),
            "optimal_calls":        res["optimal_calls"],
            "rx_at_optimal":        round(res["rx_at_optimal"], 1),
            "roi_at_optimal (%)":   round(res["roi_at_optimal"] * 100, 1),
        })
    return pd.DataFrame(rows)
