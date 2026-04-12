# utils/helpers.py
"""
Utility Functions.

Contains:
  - ISA Standard Atmosphere model (troposphere)
  - Parasite drag estimation (flat-plate analogy)
  - Plotting functions for aerodynamic results

Atmosphere model implements the International Standard Atmosphere (ISA)
with Sutherland's law for dynamic viscosity.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict
from core.config import GRAVITY, R_AIR, GAMMA, T_SL, P_SL, L_LAPSE, MU_REF, T_SUTH


def get_atmosphere(altitude_m: float) -> Dict[str, float]:
    """
    ISA Standard Atmosphere Model (troposphere only, 0-11 km).

    Implements:
      - Temperature: T = T_SL - L * h
      - Pressure: p = p_SL * (T/T_SL)^(g/(L*R))
      - Density: ρ = p / (R * T)
      - Viscosity: Sutherland's law
      - Speed of sound: a = sqrt(γ * R * T)

    Args:
        altitude_m: Altitude in meters (clamped to [0, 11000]).

    Returns:
        Dictionary with keys: rho, t, p, mu, a
    """
    # Clamp to troposphere
    alt = np.clip(altitude_m, 0.0, 11000.0)

    # Temperature
    t = T_SL - L_LAPSE * alt

    # Pressure (barometric formula)
    p = P_SL * (t / T_SL) ** (GRAVITY / (L_LAPSE * R_AIR))

    # Density (ideal gas)
    rho = p / (R_AIR * t)

    # Dynamic viscosity (Sutherland's law)
    mu = MU_REF * (t / 273.15) ** 1.5 * (273.15 + T_SUTH) / (t + T_SUTH)

    # Speed of sound
    a = np.sqrt(GAMMA * R_AIR * t)

    return {
        'rho': float(rho),
        't': float(t),
        'p': float(p),
        'mu': float(mu),
        'a': float(a),
    }


def estimate_parasite_drag(
    rho: float,
    V: float,
    root_chord: float,
    tip_chord: float,
    span: float,
    mu: float,
    thickness: float,
    thickness_loc: float,
) -> float:
    """
    Estimate parasite drag coefficient (CD0) using flat-plate analogy.

    Method: Raymer's component buildup (simplified for wing only).
      - Skin friction: Cf = 0.455 / (log10(Re_MAC))^2.58 (turbulent)
      - Form factor: FF = 1 + 0.6/x_c * t/c + 100*(t/c)^4
      - Interference factor: Q ≈ 1.0 (wing-only, no fuselage)
      - CD0 = Cf * FF * Q * (S_wet / S_ref)
      - S_wet / S_ref ≈ 2.0 for a clean wing

    Args:
        rho: Air density (kg/m³).
        V: Freestream velocity (m/s).
        root_chord: Root chord (m).
        tip_chord: Tip chord (m).
        span: Wingspan (m).
        mu: Dynamic viscosity (Pa·s).
        thickness: Max thickness ratio (t/c).
        thickness_loc: Chordwise location of max thickness (x/c).

    Returns:
        Estimated CD0.
    """
    c_mac = (root_chord + tip_chord) / 2.0
    Re_mac = rho * V * c_mac / mu

    # Turbulent flat-plate skin friction (Schlichting)
    Re_mac = max(Re_mac, 1e5)  # Floor to avoid log(0)
    Cf = 0.455 / (np.log10(Re_mac)) ** 2.58

    # Form factor (Raymer)
    x_c = max(thickness_loc, 0.05)  # Avoid division by zero
    FF = 1.0 + (0.6 / x_c) * thickness + 100.0 * thickness ** 4

    # Wetted area ratio (wing only, both surfaces)
    S_wet_ratio = 2.0 * (1.0 + 0.2 * thickness)  # Slight increase for thick airfoils

    # Interference factor
    Q = 1.0

    cd0 = Cf * FF * Q * S_wet_ratio

    # Sanity clamp
    cd0 = np.clip(cd0, 0.003, 0.05)

    return float(cd0)


def plot_aerodynamics(results: dict, output_dir: str = "results") -> None:
    """
    Generate publication-quality aerodynamic plots.

    Creates:
      1. CL vs Alpha (lift curve)
      2. Drag Polar (CL vs CD)
      3. Spanwise Lift Distribution
      4. L/D vs Alpha (efficiency)

    Args:
        results: Dictionary with keys:
            'aoa_sweep', 'cl_sweep', 'cd_sweep',
            'y_dist' (optional), 'cl_dist' (optional)
        output_dir: Directory to save plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('bmh')
    fig_dpi = 300

    aoa = results['aoa_sweep']
    cl = results['cl_sweep']
    cd = results['cd_sweep']

    # --- 1. CL vs Alpha ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(aoa, cl, 'o-', linewidth=2, color='#1D4ED8',
            markersize=4, label='Wing CL (LLT + ML)')
    ax.set_xlabel('Angle of Attack (deg)', fontsize=12)
    ax.set_ylabel('Lift Coefficient (CL)', fontsize=12)
    ax.set_title('Wing Lift Curve', fontsize=14, fontweight='bold', pad=15)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(output_dir, 'cl_vs_alpha.png'),
                dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Drag Polar ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cd, cl, 'o-', linewidth=2, color='#059669',
            markersize=4, label='Wing Polar')
    ax.set_xlabel('Drag Coefficient (CD)', fontsize=12)
    ax.set_ylabel('Lift Coefficient (CL)', fontsize=12)
    ax.set_title('Wing Drag Polar', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(output_dir, 'drag_polar.png'),
                dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)

    # --- 3. Spanwise Distribution ---
    if results.get('y_dist') is not None and results.get('cl_dist') is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_dist = results['y_dist']
        cl_dist = results['cl_dist']

        # Mirror to show full span
        y_full = np.concatenate([-y_dist[::-1], y_dist])
        cl_full = np.concatenate([cl_dist[::-1], cl_dist])

        ax.plot(y_full, cl_full, linewidth=2.5, color='#DC2626', label='Local Cl')
        ax.fill_between(y_full, cl_full, alpha=0.15, color='#DC2626')
        ax.set_xlabel('Spanwise Position (m)', fontsize=12)
        ax.set_ylabel('Local Lift Coefficient (Cl)', fontsize=12)
        ax.set_title('Spanwise Lift Distribution', fontsize=14,
                      fontweight='bold', pad=15)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.savefig(os.path.join(output_dir, 'spanwise_lift.png'),
                    dpi=fig_dpi, bbox_inches='tight')
        plt.close(fig)

    # --- 4. L/D vs Alpha ---
    ld = cl / cd
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(aoa, ld, 'o-', linewidth=2, color='#7C3AED',
            markersize=4, label='L/D')
    best_idx = np.argmax(ld)
    ax.plot(aoa[best_idx], ld[best_idx], 'r*', markersize=15,
            label=f'Best L/D = {ld[best_idx]:.1f} @ {aoa[best_idx]:.0f}°')
    ax.set_xlabel('Angle of Attack (deg)', fontsize=12)
    ax.set_ylabel('Lift/Drag Ratio', fontsize=12)
    ax.set_title('Aerodynamic Efficiency', fontsize=14,
                  fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(output_dir, 'ld_ratio.png'),
                dpi=fig_dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")
