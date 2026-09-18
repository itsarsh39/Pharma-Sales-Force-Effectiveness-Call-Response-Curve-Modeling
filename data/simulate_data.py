"""
simulate_data.py — Synthetic Dataset Generation for Pharma SFE Modeling
========================================================================

ASSUMPTIONS & RATIONALE
-------------------------
1. Response Curve Shape — Michaelis-Menten / Saturation Model
   Rx(calls) = (Vmax * calls) / (Km + calls)

   This functional form is borrowed from enzyme kinetics (Michaelis & Menten, 1913)
   and is widely used in pharma detailing literature to model the saturation effect:
   early calls educate the physician and drive prescription lift; later calls yield
   progressively smaller incremental benefit as the physician has already decided
   whether to prescribe. This contrasts with a simple log curve which can grow
   unboundedly; the MM curve has a hard asymptote (Vmax) representing the maximum
   realistic prescribing rate given the doctor's patient load and practice style.

   Reference: Sood, A. & Chen, Y. (2012). "Optimal Detailing and Sampling Policies
   in Pharmaceutical Markets." Marketing Science.

2. Territory Random Effects
   Each of the 20 territories has a random intercept drawn from N(0, sigma_territory).
   This models unobserved territory-level heterogeneity (e.g., competitor presence,
   regional formulary status, payer mix) that inflates/deflates baseline Rx
   independent of call activity.

3. Doctor Specialty as a Moderating Factor
   Specialty affects the Vmax parameter — i.e., the ceiling Rx volume achievable.
   GPs have modest Vmax (broad patient panel, low average Rx per patient per month);
   Endocrinologists and Cardiologists have higher Vmax (disease specialist, higher
   Rx propensity for the target drug class).

4. Noise Model
   Residual noise is multiplicative (lognormal), not additive, because Rx counts
   are non-negative and variance tends to scale with the mean — consistent with
   a negative binomial / overdispersed Poisson data-generating process in reality.
   We approximate this as: observed_Rx = true_Rx * exp(epsilon), epsilon ~ N(0, sigma_noise).
   Clipped at 0 to preserve non-negativity.

5. Call Count Distribution
   Monthly calls per doctor-rep pair ~ Poisson(lambda), where lambda differs by
   specialty (specialists receive more calls per industry norm) and is clipped to
   [1, 12]. This mirrors the "call-plan" constraint reps operate under.

DATASET DIMENSIONS
------------------
- 20 territories
- ~30 doctors per territory = 600 doctors
- 5 reps per territory = 100 reps
- 12 months of panel data
- Total rows ≈ 600 × 12 = 7,200 doctor-month observations
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────
# Simulation Parameters (all documented)
# ─────────────────────────────────────────────

# Number of territories, doctors per territory, reps per territory, months
N_TERRITORIES = 20
DOCTORS_PER_TERRITORY = 30
REPS_PER_TERRITORY = 5
N_MONTHS = 12

# Territory random-effect standard deviation
# Interpretation: ±1 SD shift in territory baseline ≈ ±8 Rx/month
SIGMA_TERRITORY = 8.0

# Residual noise (log-scale SD)
# Interpretation: ~25% coefficient of variation around the mean Rx prediction
SIGMA_NOISE = 0.25

# Doctor specialties and their Michaelis-Menten Vmax multipliers
# GP: moderate volume, moderate response ceiling
# Cardiologist: high Rx ceiling for cardiovascular drug analogue
# Endocrinologist: high Rx ceiling for metabolic drug analogue
# Neurologist: moderate-high ceiling
SPECIALTIES = {
    "GP":               {"vmax_mult": 1.0,  "call_lambda": 3.0},
    "Cardiologist":     {"vmax_mult": 1.8,  "call_lambda": 5.0},
    "Endocrinologist":  {"vmax_mult": 2.1,  "call_lambda": 5.5},
    "Neurologist":      {"vmax_mult": 1.5,  "call_lambda": 4.0},
}

# Base Michaelis-Menten parameters (BEFORE specialty modifier)
# Vmax_base: asymptotic Rx volume for a GP (units: Rx/month)
# Km: half-saturation constant — number of calls at which Rx = Vmax/2
#     Km=4 means at 4 calls/month a GP reaches half their prescribing ceiling
VMAX_BASE = 25.0   # Rx/month for GP at saturation
KM = 4.0            # calls/month at half-saturation

# ─────────────────────────────────────────────
# Helper: Michaelis-Menten response function
# ─────────────────────────────────────────────

def michaelis_menten(calls: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """
    Computes expected Rx volume given call count using the Michaelis-Menten
    saturation model:  Rx = (Vmax * calls) / (Km + calls)

    Parameters
    ----------
    calls : array of call counts (non-negative)
    vmax  : asymptotic maximum Rx (specialty-specific)
    km    : half-saturation constant (calls at which Rx = Vmax/2)

    Returns
    -------
    Rx_expected : array of expected Rx volumes
    """
    return (vmax * calls) / (km + calls)


# ─────────────────────────────────────────────
# Build Entity Tables
# ─────────────────────────────────────────────

def build_territory_table() -> pd.DataFrame:
    """Create territory-level metadata with random intercepts."""
    territory_ids = [f"T{str(i).zfill(2)}" for i in range(1, N_TERRITORIES + 1)]
    # Random intercepts: some territories are systematically higher/lower baseline
    territory_re = rng.normal(0, SIGMA_TERRITORY, size=N_TERRITORIES)
    return pd.DataFrame({
        "territory_id": territory_ids,
        "territory_re": territory_re,   # stored for reference / truth comparison
    })


def build_doctor_table(territory_df: pd.DataFrame) -> pd.DataFrame:
    """Create doctor-level metadata (specialty, territory assignment)."""
    specialty_list = list(SPECIALTIES.keys())
    records = []
    doc_counter = 1
    rep_counter = 1

    for _, terr_row in territory_df.iterrows():
        territory_id = terr_row["territory_id"]

        # Assign reps to this territory
        rep_ids = [f"R{str(rep_counter + j).zfill(3)}" for j in range(REPS_PER_TERRITORY)]
        rep_counter += REPS_PER_TERRITORY

        for i in range(DOCTORS_PER_TERRITORY):
            specialty = specialty_list[i % len(specialty_list)]  # cycle through specialties
            doctor_id = f"DR{str(doc_counter).zfill(4)}"
            # Each doctor is "owned" by one rep (simplified: round-robin assignment)
            rep_id = rep_ids[i % REPS_PER_TERRITORY]
            records.append({
                "doctor_id": doctor_id,
                "territory_id": territory_id,
                "rep_id": rep_id,
                "doctor_specialty": specialty,
            })
            doc_counter += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Build Panel Dataset (doctor × month)
# ─────────────────────────────────────────────

def simulate_panel(doctor_df: pd.DataFrame, territory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a panel of (doctor, month) observations with simulated call counts
    and Rx volumes that follow the Michaelis-Menten curve with territory-level
    random effects and multiplicative noise.
    """
    territory_re_map = territory_df.set_index("territory_id")["territory_re"].to_dict()

    records = []
    for _, doc in doctor_df.iterrows():
        spec_params = SPECIALTIES[doc["doctor_specialty"]]
        vmax = VMAX_BASE * spec_params["vmax_mult"]
        call_lambda = spec_params["call_lambda"]
        terr_re = territory_re_map[doc["territory_id"]]

        for month in range(1, N_MONTHS + 1):
            # ── Call count: Poisson with specialty-specific mean, clipped [1, 12]
            calls = int(np.clip(rng.poisson(call_lambda), 1, 12))

            # ── True expected Rx: MM curve + territory random effect
            true_rx_mean = michaelis_menten(calls, vmax, KM) + terr_re
            true_rx_mean = max(true_rx_mean, 1.0)  # floor at 1 to keep log-space valid

            # ── Multiplicative noise (lognormal)
            epsilon = rng.normal(0, SIGMA_NOISE)
            observed_rx = true_rx_mean * np.exp(epsilon)
            observed_rx = max(round(observed_rx, 1), 0.0)  # non-negative, 1dp

            records.append({
                "rep_id":            doc["rep_id"],
                "territory_id":      doc["territory_id"],
                "doctor_id":         doc["doctor_id"],
                "doctor_specialty":  doc["doctor_specialty"],
                "month":             month,
                "monthly_call_count": calls,
                "rx_volume":         observed_rx,
                # Ground-truth columns (kept for model validation only)
                "_true_vmax":        vmax,
                "_true_km":          KM,
                "_true_territory_re": terr_re,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Simulating pharma SFE dataset...")

    territory_df = build_territory_table()
    doctor_df    = build_doctor_table(territory_df)
    panel_df     = simulate_panel(doctor_df, territory_df)

    print(f"  Territories : {N_TERRITORIES}")
    print(f"  Doctors     : {len(doctor_df)}")
    print(f"  Total rows  : {len(panel_df)}")
    print(f"  Call range  : {panel_df['monthly_call_count'].min()}–{panel_df['monthly_call_count'].max()}")
    print(f"  Rx range    : {panel_df['rx_volume'].min():.1f}–{panel_df['rx_volume'].max():.1f}")

    # Save outputs
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path    = out_dir / "panel_data.csv"
    territory_path = out_dir / "territory_metadata.csv"
    doctor_path   = out_dir / "doctor_metadata.csv"

    panel_df.to_csv(panel_path, index=False)
    territory_df.to_csv(territory_path, index=False)
    doctor_df.to_csv(doctor_path, index=False)

    print(f"\nSaved:\n  {panel_path}\n  {territory_path}\n  {doctor_path}")
    return panel_df, territory_df, doctor_df


if __name__ == "__main__":
    main()
