# main.py
"""
Wing Analyzer — Deterministic Entry Point.

Pipeline:
  1. Load configuration (preset or custom)
  2. Compute atmosphere at cruise altitude
  3. Load ML models for airfoil prediction
  4. Extract section aerodynamic properties (a0, α_L0) via ML
  5. Run Lifting Line Theory across AoA sweep
  6. Apply parasite drag estimation
  7. Validate all results
  8. Print summary + save plots

No Streamlit. No interactive UI. Pure analysis pipeline.
"""

import numpy as np
import os
import sys

from core.airfoil_ml import AirfoilML
from core.physics import enforce_physics
from core.llt import solve_llt, LLTError
from core.validation import (
    validate_wing_coefficients,
    validate_spanwise_distribution,
    validate_cross_check,
)
from core.config import PRESETS, get_preset
from utils.helpers import get_atmosphere, estimate_parasite_drag, plot_aerodynamics


def extract_section_properties(
    ml_model: AirfoilML,
    re_stations: np.ndarray,
    config: dict,
    N: int,
) -> tuple:
    """
    Extract section lift-curve slope (a0) and zero-lift angle (α_L0)
    at each spanwise station using ML model predictions.

    Method:
      - For each station, predict Cl at multiple AoA in the linear range
      - Fit a linear regression to get slope (a0) and intercept
      - Compute zero-lift angle from intercept

    Args:
        ml_model: Loaded AirfoilML instance.
        re_stations: Reynolds numbers at each station.
        config: Wing configuration dict.
        N: Number of stations.

    Returns:
        Tuple of (a0_dist, al0_dist) — both arrays of length N.
        a0_dist: section lift-curve slopes (per radian)
        al0_dist: section zero-lift angles (radians)
    """
    a0_dist = np.empty(N)
    al0_dist = np.empty(N)

    # Sample AoA range in the linear region only (-5° to +5°)
    aoa_samples = np.arange(-5.0, 6.0, 1.0)
    aoa_samples_rad = np.radians(aoa_samples)

    for i in range(N):
        cl_samples = np.empty(len(aoa_samples))

        for k, aoa in enumerate(aoa_samples):
            cl, _ = ml_model.predict(
                re_stations[i], aoa,
                config['thickness'], config['thickness_loc'],
                config['camber'], config['camber_loc'],
            )
            # Apply physics constraints to each sample
            cl, _ = enforce_physics(
                aoa, cl, 0.01,  # Cd doesn't matter here
                config['thickness'], config['camber'],
            )
            cl_samples[k] = cl

        # Linear fit: Cl = a0 * alpha_rad + b
        coeffs = np.polyfit(aoa_samples_rad, cl_samples, 1)
        a0 = coeffs[0]  # lift-curve slope (per radian)
        b = coeffs[1]   # Cl at alpha=0

        # Zero-lift angle: 0 = a0 * al0 + b → al0 = -b / a0
        if abs(a0) > 0.1:  # Sanity: slope should be meaningfully positive
            al0 = -b / a0
        else:
            # Fallback: use thin-airfoil estimate
            al0 = -2.0 * config['camber']
            a0 = 2.0 * np.pi  # Thin-airfoil default

        a0_dist[i] = a0
        al0_dist[i] = al0

    return a0_dist, al0_dist


def run_analysis(wing_type: str = 'general_aviation', N: int = 40) -> dict:
    """
    Run complete wing aerodynamic analysis.

    Args:
        wing_type: Preset name from config (or custom dict).
        N: Number of spanwise stations.

    Returns:
        Results dictionary with sweep data and distributions.
    """
    print(f"\n{'='*60}")
    print(f"  WING ANALYZER — {wing_type.upper()}")
    print(f"{'='*60}")

    # --- 1. Configuration ---
    config = get_preset(wing_type)
    print(f"\n[1/7] Configuration: {config['label']}")
    print(f"      Span={config['span']}m, "
          f"Root={config['root_chord']}m, Tip={config['tip_chord']}m")

    # --- 2. Atmosphere ---
    atm = get_atmosphere(config['altitude'])
    rho, mu, V = atm['rho'], atm['mu'], config['cruise_speed']
    mach = V / atm['a']
    print(f"\n[2/7] Atmosphere @ {config['altitude']}m:")
    print(f"      rho={rho:.4f} kg/m3, mu={mu:.2e} Pa.s, "
          f"V={V} m/s, Mach={mach:.3f}")

    if mach > 0.3:
        print(f"      [!] Mach {mach:.3f} > 0.3 -- compressibility effects not modeled")

    # --- 3. Load ML model ---
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ml_model = AirfoilML(models_dir)
    print(f"\n[3/7] ML models loaded from {models_dir}")

    # --- 4. Spanwise discretization + section properties ---
    theta = np.linspace(np.pi / (2 * N), np.pi / 2, N)
    y_stations = (config['span'] / 2) * np.cos(theta)
    eta_stations = y_stations / (config['span'] / 2)
    chord_stations = (config['root_chord']
                      + (config['tip_chord'] - config['root_chord']) * (1 - eta_stations))
    re_stations = rho * V * chord_stations / mu

    print(f"\n[4/7] Extracting section properties ({N} stations)...")
    print(f"      Re range: [{re_stations.min():.0f}, {re_stations.max():.0f}]")

    a0_dist, al0_dist = extract_section_properties(ml_model, re_stations, config, N)

    print(f"      a0 range:  [{a0_dist.min():.2f}, {a0_dist.max():.2f}] /rad "
          f"(theory: {2*np.pi:.2f})")
    print(f"      aL0 range: [{np.degrees(al0_dist.min()):.2f} deg, "
          f"{np.degrees(al0_dist.max()):.2f} deg]")

    # --- 5. AoA Sweep using LLT ---
    aoa_sweep = np.arange(-4.0, 16.0, 1.0)
    cl_results = []
    cdi_results = []
    e_results = []
    sample_cl_dist = None
    sample_y_dist = None
    sample_alpha_i = None

    print(f"\n[5/7] Running LLT sweep (alpha = {aoa_sweep[0]:.0f} to {aoa_sweep[-1]:.0f} deg)...")

    failed_count = 0
    for aoa in aoa_sweep:
        try:
            res = solve_llt(
                N, aoa, config['span'],
                config['root_chord'], config['tip_chord'],
                config['twist_deg'], a0_dist, al0_dist,
            )
        except LLTError as err:
            print(f"      LLT error at alpha={aoa:.0f} deg: {err}")
            failed_count += 1
            continue

        if res is None:
            failed_count += 1
            continue

        # Log warnings
        for w in res.warnings:
            print(f"      [LLT α={aoa:.0f}°] {w}")

        cl_results.append(res.CL)
        cdi_results.append(res.CDi)
        e_results.append(res.e)

        # Capture distribution at 5° for plotting
        if abs(aoa - 5.0) < 0.5:
            sample_cl_dist = res.cl_dist
            sample_y_dist = res.y_dist
            sample_alpha_i = res.alpha_i_dist

    if failed_count > 0:
        print(f"      [!] {failed_count}/{len(aoa_sweep)} AoA points failed")

    # Trim aoa_sweep to match successful results
    aoa_valid = aoa_sweep[:len(cl_results)]
    cl_array = np.array(cl_results)
    cdi_array = np.array(cdi_results)

    # --- 6. Parasite drag ---
    cd0 = estimate_parasite_drag(
        rho, V, config['root_chord'], config['tip_chord'],
        config['span'], mu, config['thickness'], config['thickness_loc'],
    )
    cd_total = cdi_array + cd0

    print(f"\n[6/7] Drag estimation:")
    print(f"      CD0 (parasite) = {cd0:.5f}")
    print(f"      CDi range:  [{cdi_array.min():.5f}, {cdi_array.max():.5f}]")
    print(f"      Oswald e:  [{min(e_results):.3f}, {max(e_results):.3f}]")

    # --- 7. Validation ---
    print(f"\n[7/7] Validation:")

    max_cl = float(cl_array.max())
    max_cd = float(cd_total.max())
    best_ld_idx = np.argmax(cl_array / cd_total)
    best_ld = cl_array[best_ld_idx] / cd_total[best_ld_idx]
    best_ld_aoa = aoa_valid[best_ld_idx]

    # Wing coefficient validation
    report = validate_wing_coefficients(
        max_cl, max_cd,
        CDi=cdi_array.max(),
        e=min(e_results) if e_results else None,
        AR=config['span']**2 / (0.5 * (config['root_chord'] + config['tip_chord']) * config['span']),
    )
    print(f"      {report.summary()}")

    # Spanwise distribution validation
    if sample_cl_dist is not None and sample_y_dist is not None:
        dist_report = validate_spanwise_distribution(
            sample_y_dist, sample_cl_dist, config['span']
        )
        if dist_report.issues:
            print(f"      Spanwise: {dist_report.summary()}")

    # Cross-check at 5°
    if sample_cl_dist is not None:
        cross = validate_cross_check(
            cl_array[np.argmin(np.abs(aoa_valid - 5.0))],
            5.0,
            config['span']**2 / (0.5 * (config['root_chord'] + config['tip_chord']) * config['span']),
        )
        if cross.issues:
            print(f"      Cross-check: {cross.summary()}")

    # --- Summary ---
    print(f"\n{'-'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'-'*60}")
    print(f"  Max CL:     {max_cl:.4f}")
    print(f"  CD0:        {cd0:.5f}")
    print(f"  Max CD:     {max_cd:.5f}")
    print(f"  Best L/D:   {best_ld:.1f} @ alpha={best_ld_aoa:.0f} deg")
    print(f"  Valid:      {'PASS' if report.is_valid else 'FAIL'}")
    print(f"{'-'*60}\n")

    # --- Package results ---
    results = {
        'wing_type': wing_type,
        'config': config,
        'aoa_sweep': aoa_valid,
        'cl_sweep': cl_array,
        'cd_sweep': cd_total,
        'cdi_sweep': cdi_array,
        'cd0': cd0,
        'y_dist': sample_y_dist,
        'cl_dist': sample_cl_dist,
        'alpha_i_dist': sample_alpha_i,
        'best_ld': best_ld,
        'best_ld_aoa': best_ld_aoa,
        'validation': report,
    }

    # --- Plot ---
    plot_aerodynamics(results)

    return results


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    # Run for default preset
    preset = sys.argv[1] if len(sys.argv) > 1 else 'general_aviation'
    run_analysis(preset)
