"""
modeling.py — Reusable Modeling Functions for Pharma SFE Analysis
==================================================================

Provides three modeling layers, each building on the previous:

1. baseline_ols()
   Simple OLS (Rx ~ calls). Deliberately naive — used to demonstrate linear
   model misspecification (curvature in residuals, heteroscedasticity) and
   motivate non-linear alternatives.

2. fit_response_curve()
   Fits a Michaelis-Menten saturation curve per specialty/segment using
   scipy.optimize.curve_fit with bounded parameters. Returns fitted Vmax
   and Km with 95% CIs from the covariance matrix.

3. fit_mixed_lm()
   Fits a log-transform linear mixed model via statsmodels.MixedLM:
       log(Rx) ~ log(calls) + specialty_dummies + (1 | territory)
   WHY MIXED-EFFECTS? Doctors within the same territory share unobserved
   factors (formulary status, regional disease prevalence, competitor
   activity). Ignoring this cluster structure:
   (a) underestimates standard errors → false statistical confidence
   (b) biases coefficient estimates if territory effects correlate with
       call allocation (which they will, since reps target territories
       with higher commercial potential).
   Random intercepts by territory correctly partial out this between-
   territory variation, leaving the fixed-effect call coefficient to
   capture the within-territory, within-doctor response to detailing.

DESIGN NOTES
------------
- All functions accept a pd.DataFrame and return a results dict for easy
  downstream use by optimization.py and the Streamlit app.
- AIC/BIC are computed uniformly for model comparison.
- Residuals are returned alongside predictions so the notebook can run
  diagnostics without duplicating code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE OLS
# ─────────────────────────────────────────────────────────────────────────────

def baseline_ols(df: pd.DataFrame,
                 outcome: str = "rx_volume",
                 predictor: str = "monthly_call_count") -> dict[str, Any]:
    """
    Fits a simple OLS: outcome ~ predictor.

    Parameters
    ----------
    df        : panel dataframe
    outcome   : column name for Rx volume
    predictor : column name for call count

    Returns
    -------
    dict with keys:
        'model'       : fitted OLS result object
        'predictions' : array of fitted values
        'residuals'   : array of residuals
        'aic'         : Akaike Information Criterion
        'bic'         : Bayesian Information Criterion
        'rsq'         : R-squared
        'formula'     : string formula used
    """
    formula = f"{outcome} ~ {predictor}"
    model = smf.ols(formula=formula, data=df).fit()

    return {
        "model":       model,
        "predictions": model.fittedvalues.values,
        "residuals":   model.resid.values,
        "aic":         model.aic,
        "bic":         model.bic,
        "rsq":         model.rsquared,
        "formula":     formula,
        "name":        "Baseline OLS",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. NON-LINEAR RESPONSE CURVE (Michaelis-Menten)
# ─────────────────────────────────────────────────────────────────────────────

def _mm_curve(calls: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """Michaelis-Menten function for curve_fit."""
    return (vmax * calls) / (km + calls)


def fit_response_curve(df: pd.DataFrame,
                       group_col: str | None = None,
                       call_col: str = "monthly_call_count",
                       rx_col: str = "rx_volume",
                       vmax_init: float = 40.0,
                       km_init: float = 4.0) -> dict[str, Any]:
    """
    Fits a Michaelis-Menten saturation curve, optionally per group
    (e.g., doctor_specialty or territory_id).

    WHY MM CURVE?
    A log(x+1) transformation is simpler but doesn't have a natural ceiling.
    The MM form has:
      - Rx → 0 as calls → 0 (correct: no detailing, no product awareness)
      - Rx → Vmax as calls → ∞ (correct: biological/time ceiling on prescribing)
      - One interpretable parameter (Km) capturing "calls at half-saturation"
    This interpretability is crucial for translating to call-plan recommendations.

    Parameters
    ----------
    df        : dataframe
    group_col : if not None, fit separately per unique value of this column
    call_col  : column for monthly call count
    rx_col    : column for Rx volume
    vmax_init : initial guess for Vmax parameter
    km_init   : initial guess for Km parameter

    Returns
    -------
    dict with:
        'params'   : dict (or dict-of-dicts if group_col provided)
        'ci_95'    : 95% confidence intervals (from covariance matrix)
        'fitted'   : array of fitted Rx values
        'residuals': residual array
        'aic'      : computed AIC
        'bic'      : computed BIC
        'name'     : label string
    """
    def _fit_group(subdf: pd.DataFrame):
        x = subdf[call_col].values.astype(float)
        y = subdf[rx_col].values.astype(float)
        try:
            popt, pcov = curve_fit(
                _mm_curve, x, y,
                p0=[vmax_init, km_init],
                bounds=([1.0, 0.1], [500.0, 50.0]),
                maxfev=10_000,
            )
            perr = np.sqrt(np.diag(pcov))
            vmax_fit, km_fit = popt
            vmax_ci = (vmax_fit - 1.96 * perr[0], vmax_fit + 1.96 * perr[0])
            km_ci   = (km_fit   - 1.96 * perr[1], km_fit   + 1.96 * perr[1])
            fitted  = _mm_curve(x, *popt)
            resid   = y - fitted

            # AIC / BIC — approximate via RSS-based log-likelihood (Gaussian noise assumed)
            n = len(y)
            rss = np.sum(resid ** 2)
            k = 2  # Vmax, Km
            sigma2_hat = rss / n
            ll = -0.5 * n * (np.log(2 * np.pi * sigma2_hat) + 1)
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)

            return {
                "vmax": vmax_fit, "km": km_fit,
                "vmax_ci": vmax_ci, "km_ci": km_ci,
                "fitted": fitted, "residuals": resid,
                "aic": aic, "bic": bic,
                "n": n, "converged": True,
            }
        except RuntimeError:
            warnings.warn(f"curve_fit did not converge for group.")
            return {"converged": False}

    if group_col is None:
        result = _fit_group(df)
        result["name"] = "MM Response Curve (pooled)"
        return result

    # Per-group fitting
    groups = {}
    all_fitted = np.zeros(len(df))
    all_resid  = np.zeros(len(df))
    df = df.reset_index(drop=True)

    for grp_val, subdf in df.groupby(group_col):
        idx = subdf.index
        res = _fit_group(subdf)
        groups[grp_val] = res
        if res.get("converged", False):
            all_fitted[idx] = res["fitted"]
            all_resid[idx]  = res["residuals"]

    # Overall AIC/BIC (sum of group AICs — valid since groups are disjoint)
    total_aic = sum(g["aic"] for g in groups.values() if g.get("converged"))
    total_bic = sum(g["bic"] for g in groups.values() if g.get("converged"))

    return {
        "params":    groups,
        "fitted":    all_fitted,
        "residuals": all_resid,
        "aic":       total_aic,
        "bic":       total_bic,
        "name":      f"MM Response Curve (per {group_col})",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MIXED-EFFECTS MODEL
# ─────────────────────────────────────────────────────────────────────────────

def fit_mixed_lm(df: pd.DataFrame,
                 outcome: str = "rx_volume",
                 call_col: str = "monthly_call_count",
                 group_col: str = "territory_id",
                 include_specialty: bool = True) -> dict[str, Any]:
    """
    Fits a linear mixed-effects model via statsmodels MixedLM.

    Model specification:
        log(rx_volume + 1) ~ log(monthly_call_count) [+ specialty dummies]
                           + (1 | territory_id)

    WHY LOG TRANSFORMATION?
    - Rx counts are right-skewed and non-negative.
    - Log-log (or log-linear) specification gives elasticity interpretation:
      coefficient β on log(calls) ≈ % change in Rx per 1% increase in calls.
    - Combined with random intercepts, this is equivalent to a log-linear
      multilevel model, standard in health economics panel data analysis.

    WHY RANDOM INTERCEPTS (NOT SLOPES)?
    Allowing random slopes (territory-specific call elasticities) would be
    ideal but requires N_docs >> N_territories for stable estimation.
    With 30 docs/territory we have marginal power; random intercepts capture
    the most important source of heterogeneity (baseline Rx differences)
    without overfitting.

    Parameters
    ----------
    df               : panel dataframe
    outcome          : Rx volume column
    call_col         : call count column
    group_col        : grouping column for random effects (territory)
    include_specialty: whether to include specialty fixed effects

    Returns
    -------
    dict with model result, AIC/BIC, predictions, residuals
    """
    data = df.copy()
    data["log_rx"]    = np.log1p(data[outcome])
    data["log_calls"] = np.log(data[call_col].clip(lower=1))

    if include_specialty:
        formula = "log_rx ~ log_calls + C(doctor_specialty)"
    else:
        formula = "log_rx ~ log_calls"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula=formula, data=data, groups=data[group_col])
        result = model.fit(reml=True, method="lbfgs")

    fitted    = result.fittedvalues.values
    residuals = result.resid.values

    # AIC/BIC: statsmodels MixedLM with REML returns NaN for .aic / .bic
    # because REML log-likelihood is not comparable to ML. We use the ML
    # approximation: re-fit with method="ml" for IC purposes only.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_ml = model.fit(reml=False, method="lbfgs")
        aic_val = result_ml.aic
        bic_val = result_ml.bic
    except Exception:
        # Fallback: compute from REML log-likelihood
        ll   = result.llf
        k    = len(result.params)
        n    = len(data)
        aic_val = -2 * ll + 2 * k
        bic_val = -2 * ll + k * np.log(n)

    return {
        "model":          result,
        "predictions":    fitted,
        "residuals":      residuals,
        "aic":            aic_val,
        "bic":            bic_val,
        "log_scale":      True,    # flag: predictions are in log(Rx+1) space
        "formula":        formula,
        "name":           "Mixed LM (territory RE + specialty FE)",
        "random_effects": result.random_effects,  # territory-level intercepts
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. DIAGNOSTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Computes Variance Inflation Factors for a set of features.
    VIF > 10 indicates problematic multicollinearity.
    VIF > 5 warrants investigation.

    Parameters
    ----------
    df           : dataframe containing feature_cols
    feature_cols : list of column names

    Returns
    -------
    DataFrame with columns ['Feature', 'VIF']
    """
    X = sm.add_constant(df[feature_cols].dropna())
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)


def residual_diagnostics(residuals: np.ndarray,
                         fitted: np.ndarray) -> dict[str, Any]:
    """
    Runs a battery of residual diagnostics:
    - Shapiro-Wilk normality test (sample ≤ 5000, else skipped)
    - Mean and std of residuals
    - Breusch-Pagan-style check (correlation of |resid| with fitted)

    Parameters
    ----------
    residuals : array of model residuals
    fitted    : array of fitted values

    Returns
    -------
    dict of diagnostic results
    """
    results: dict[str, Any] = {}

    results["mean_resid"] = float(np.mean(residuals))
    results["std_resid"]  = float(np.std(residuals))

    # Normality (Shapiro-Wilk — valid up to n=5000)
    n = len(residuals)
    if n <= 5000:
        stat, p = shapiro(residuals)
        results["shapiro_stat"] = float(stat)
        results["shapiro_p"]    = float(p)
        results["normality_ok"] = bool(p > 0.05)
    else:
        results["shapiro_note"] = "Skipped (n > 5000); use Q-Q plot for normality check."

    # Heteroscedasticity proxy: Pearson r between |residuals| and fitted values
    corr = np.corrcoef(np.abs(residuals), fitted)[0, 1]
    results["hetero_corr"]    = float(corr)
    results["hetero_concern"] = abs(corr) > 0.15

    return results
