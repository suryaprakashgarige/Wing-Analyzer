# main.py
"""
Wing Analyzer -- Deterministic Entry Point.

Pipeline (sequential, controlled):
  1. Load configuration
  2. Compute atmosphere
  3. Load ML model
  4. OOD check on inputs
  5. ML predict -> physics clamp -> lift slope extraction (a0, alpha_L0)
  6. LLT solver across AoA sweep
  7. Drag model: CD = CD0 + CDi (not ML Cd)
  8. Validation
  9. Print summary + save plots

No Streamlit. No interactive UI. Pure analysis pipeline.
"""

import numpy as np
import os
import sys

from core.airfoil_ml import AirfoilML, compute_lift_slope, compute_zero_lift_angle
from core.physics import enforce_physics
from core.llt import solve_llt, LLTError
from core.drag import compute_wing_drag, compute_form_factor
from core.ood import check_prediction_ood
from core.validation import (
    validate_wing_coefficients,
    validate_spanwise_distribution,
    validate_cross_check,
)
from core.config import PRESETS, get_preset
from utils.helpers import get_atmosphere, plot_aerodynamics


def extract_section_properties(
    ml_model: AirfoilML,
    re_stations: np.ndarray,
    config: dict,
    N: int,
) -> tuple:
    """
    Extract section lift-curve slope (a0) and zero-lift angle (alpha_L0)
    at each spanwise station.

    Pipeline per station:
      1. OOD check
      2. compute_lift_slope() -- central difference from ML, clamped [4.5, 7.0]
      3. compute_zero_lift_angle() -- interpolation from ML
      4. If OOD: use thin-airfoil fallback

    Args:
        ml_model: Loaded AirfoilML instance.
        re_stations: Reynolds numbers at each station.
        config: Wing configuration dict.
        N: Number of stations.

    Returns:
        Tuple of (a0_dist, al0_dist, ood_count).
        a0_dist: section lift-curve slopes (per radian), clamped [4.5, 7.0]
        al0_dist: section zero-lift angles (radians)
        ood_count: number of stations flagged as out-of-distribution
    """
    a0_dist = np.empty(N)
    al0_dist = np.empty(N)
    ood_count = 0

    for i in range(N):
        # --- Step 1: OOD check ---
        ood = check_prediction_ood(
            re_stations[i], 0.0,  # Check at alpha=0 (representative)
            config['thickness'], config['thickness_loc'],
            config['camber'], config['camber_loc'],
        )

        if not ood.is_in_distribution:
            # Fallback to thin-airfoil theory
            a0_dist[i] = 2.0 * np.pi
            al0_dist[i] = -2.0 * config['camber']
            ood_count += 1
            continue

        # --- Step 2: Lift slope via central difference (clamped [4.5, 7.0]) ---
        a0_dist[i] = compute_lift_slope(
            ml_model, 0.0, re_stations[i],
            config['thickness'], config['thickness_loc'],
            config['camber'], config['camber_loc'],
        )

        # --- Step 3: Zero-lift angle ---
        al0_dist[i] = compute_zero_lift_angle(
            ml_model, re_stations[i],
            config['thickness'], config['thickness_loc'],
            config['camber'], config['camber_loc'],
        )

    return a0_dist, al0_dist, ood_count


def run_analysis(wing_type: str = 'general_aviation', N: int = 40) -> dict:
    """
    Run complete wing aerodynamic analysis.

    Pipeline: ML -> OOD check -> physics clamp -> lift slope -> LLT -> drag model -> validation

    Args:
        wing_type: Preset name from config.
        N: Number of spanwise stations.

    Returns:
        Results dictionary with sweep data and distributions.
    """
    print(f"\n{'='*60}")
    print(f"  WING ANALYZER -- {wing_type.upper()}")
    print(f"{'='*60}")

    # --- 1. Configuration ---
    config = get_preset(wing_type)
    print(f"\n[1/8] Configuration: {config['label']}")
    print(f"      Span={config['span']}m, "
          f"Root={config['root_chord']}m, Tip={config['tip_chord']}m")

    # --- 2. Atmosphere ---
    atm = get_atmosphere(config['altitude'])
    rho, mu, V = atm['rho'], atm['mu'], config['cruise_speed']
    mach = V / atm['a']
    print(f"\n[2/8] Atmosphere @ {config['altitude']}m:")
    print(f"      rho={rho:.4f} kg/m3, mu={mu:.2e} Pa.s, "
          f"V={V} m/s, Mach={mach:.3f}")

    if mach > 0.3:
        print(f"      [!] Mach {mach:.3f} > 0.3 -- compressibility effects not modeled")

    # --- 3. Load ML model ---
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ml_model = AirfoilML(models_dir)
    print(f"\n[3/8] ML models loaded from {models_dir}")

    # --- 4. OOD check on representative input ---
    ood_check = check_prediction_ood(
        rho * V * config['root_chord'] / mu,  # Re at root
        5.0,  # Representative AoA
        config['thickness'], config['thickness_loc'],
        config['camber'], config['camber_loc'],
    )
    print(f"\n[4/8] OOD Check:")
    if ood_check.is_in_distribution:
        print(f"      PASS -- all features within training envelope (max z={ood_check.max_z:.2f})")
    else:
        print(f"      [!] OOD detected: {', '.join(ood_check.flagged_features)}")
        print(f"      Stations outside envelope will use thin-airfoil fallback")

    # --- 5. Spanwise discretization + section properties ---
    theta = np.linspace(np.pi / (2 * N), np.pi / 2, N)
    y_stations = (config['span'] / 2) * np.cos(theta)
    eta_stations = y_stations / (config['span'] / 2)
    chord_stations = (config['root_chord']
                      + (config['tip_chord'] - config['root_chord']) * (1 - eta_stations))
    re_stations = rho * V * chord_stations / mu

    print(f"\n[5/8] Section properties ({N} stations, central-difference a0)...")
    print(f"      Re range: [{re_stations.min():.0f}, {re_stations.max():.0f}]")

    a0_dist, al0_dist, ood_stations = extract_section_properties(
        ml_model, re_stations, config, N
    )

    print(f"      a0 range:  [{a0_dist.min():.2f}, {a0_dist.max():.2f}] /rad "
          f"(theory: {2*np.pi:.2f})")
    print(f"      aL0 range: [{np.degrees(al0_dist.min()):.2f} deg, "
          f"{np.degrees(al0_dist.max()):.2f} deg]")
    if ood_stations > 0:
        print(f"      [!] {ood_stations}/{N} stations used thin-airfoil fallback (OOD)")

    # --- 6. AoA Sweep using LLT ---
    aoa_sweep = np.arange(-4.0, 16.0, 1.0)
    cl_results = []
    cdi_results = []
    e_results = []
    sample_cl_dist = None
    sample_y_dist = None
    sample_alpha_i = None

    print(f"\n[6/8] Running LLT sweep (alpha = {aoa_sweep[0]:.0f} to {aoa_sweep[-1]:.0f} deg)...")

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

        cl_results.append(res.CL)
        cdi_results.append(res.CDi)
        e_results.append(res.e)

        # Capture distribution at 5 deg for plotting
        if abs(aoa - 5.0) < 0.5:
            sample_cl_dist = res.cl_dist
            sample_y_dist = res.y_dist
            sample_alpha_i = res.alpha_i_dist

    if failed_count > 0:
        print(f"      [!] {failed_count}/{len(aoa_sweep)} AoA points failed")

    # Trim to match successful results
    aoa_valid = aoa_sweep[:len(cl_results)]
    cl_array = np.array(cl_results)
    cdi_array = np.array(cdi_results)

    # --- 7. Drag model: CD = CD0 + CDi (NOT ML Cd) ---
    AR = config['span'] ** 2 / (
        0.5 * (config['root_chord'] + config['tip_chord']) * config['span']
    )

    # Compute CD0 once (it doesn't change with AoA)
    from core.drag import compute_wing_drag
    drag_ref = compute_wing_drag(
        CL=0.5, AR=AR, e=max(e_results) if e_results else 0.9,
        rho=rho, V=V, mu=mu,
        root_chord=config['root_chord'], tip_chord=config['tip_chord'],
        span=config['span'],
        thickness=config['thickness'], thickness_loc=config['thickness_loc'],
    )
    cd0 = drag_ref.CD0

    # Total drag for each AoA point: CD = CD0 + CDi_from_LLT
    cd_total = cdi_array + cd0

    print(f"\n[7/8] Drag model (CD = CD0 + CDi):")
    print(f"      CD0 (parasite)  = {cd0:.5f} (Cf={drag_ref.Cf:.5f}, FF={drag_ref.FF:.3f})")
    print(f"      CDi range       = [{cdi_array.min():.5f}, {cdi_array.max():.5f}]")
    print(f"      CD total range  = [{cd_total.min():.5f}, {cd_total.max():.5f}]")
    print(f"      Oswald e range  = [{min(e_results):.3f}, {max(e_results):.3f}]")

    # --- 8. Validation ---
    print(f"\n[8/8] Validation:")

    max_cl = float(cl_array.max())
    max_cd = float(cd_total.max())
    best_ld_idx = np.argmax(cl_array / cd_total)
    best_ld = float(cl_array[best_ld_idx] / cd_total[best_ld_idx])
    best_ld_aoa = float(aoa_valid[best_ld_idx])

    # Wing coefficient validation
    report = validate_wing_coefficients(
        max_cl, max_cd,
        CDi=float(cdi_array.max()),
        e=min(e_results) if e_results else None,
        AR=AR,
    )
    print(f"      {report.summary()}")

    # Spanwise distribution validation
    if sample_cl_dist is not None and sample_y_dist is not None:
        dist_report = validate_spanwise_distribution(
            sample_y_dist, sample_cl_dist, config['span']
        )
        if dist_report.issues:
            print(f"      Spanwise: {dist_report.summary()}")

    # Cross-check at 5 deg
    if sample_cl_dist is not None:
        cross = validate_cross_check(
            cl_array[np.argmin(np.abs(aoa_valid - 5.0))],
            5.0, AR,
        )
        if cross.issues:
            print(f"      Cross-check: {cross.summary()}")

    # --- Summary ---
    print(f"\n{'-'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'-'*60}")
    print(f"  Max CL:     {max_cl:.4f}")
    print(f"  CD0:        {cd0:.5f}")
    print(f"  Max CDi:    {cdi_array.max():.5f}")
    print(f"  Max CD:     {max_cd:.5f}")
    print(f"  Best L/D:   {best_ld:.1f} @ alpha={best_ld_aoa:.0f} deg")
    print(f"  OOD:        {ood_stations}/{N} stations fallback")
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
        'ood_stations': ood_stations,
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
