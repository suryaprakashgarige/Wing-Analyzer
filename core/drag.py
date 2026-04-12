# core/drag.py
"""
Drag Model Module.

Computes total drag as the sum of:
  - CD0: Parasite drag (skin friction + form factor)
  - CDi: Induced drag from LLT (CL^2 / (pi * AR * e))

This replaces the fragmented drag calculations that were previously
scattered across main.py, app.py, and helpers.py.

No UI code. No plotting.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DragBreakdown:
    """Complete drag breakdown for a single operating point."""
    CD0: float       # Parasite drag coefficient
    CDi: float       # Induced drag coefficient
    CD_total: float  # CD0 + CDi
    Cf: float        # Skin friction coefficient
    FF: float        # Form factor
    Re_ref: float    # Reference Reynolds number used


def compute_cd0(
    Re: float,
    S_ref: float,
    S_wet: float,
    form_factor: float = 1.2,
) -> float:
    """
    Compute parasite drag coefficient using flat-plate skin friction.

    Method (Schlichting turbulent flat-plate correlation):
        Cf = 0.455 / (log10(Re))^2.58
        CD0 = Cf * FF * (S_wet / S_ref)

    Args:
        Re: Reynolds number based on reference length (MAC).
            Must be > 0 — clamped to minimum 1e5 internally.
        S_ref: Reference (planform) area (m^2). Must be > 0.
        S_wet: Wetted surface area (m^2). Must be > 0.
        form_factor: Airfoil form factor (default 1.2).
            Typical values: 1.1-1.4 for clean airfoils.

    Returns:
        CD0 (parasite drag coefficient). Always >= 0.003.

    Raises:
        ValueError: If S_ref or S_wet <= 0.
    """
    if S_ref <= 0:
        raise ValueError(f"S_ref must be > 0, got {S_ref}")
    if S_wet <= 0:
        raise ValueError(f"S_wet must be > 0, got {S_wet}")

    # Floor Re to avoid log(0) and unrealistically low values
    Re_safe = max(Re, 1e5)

    # Turbulent flat-plate skin friction (Schlichting)
    Cf = 0.455 / (np.log10(Re_safe)) ** 2.58

    CD0 = Cf * form_factor * (S_wet / S_ref)

    # Physical floor: no wing has less drag than ~0.003
    CD0 = max(CD0, 0.003)

    return float(CD0)


def compute_cdi(
    CL: float,
    AR: float,
    e: float,
) -> float:
    """
    Compute induced drag coefficient.

    Formula:
        CDi = CL^2 / (pi * AR * e)

    Args:
        CL: Wing lift coefficient.
        AR: Aspect ratio (span^2 / S_ref). Must be > 0.
        e: Oswald span efficiency factor (0 < e <= 1).

    Returns:
        CDi (induced drag coefficient). Always >= 0.

    Raises:
        ValueError: If AR <= 0 or e <= 0.
    """
    if AR <= 0:
        raise ValueError(f"AR must be > 0, got {AR}")
    if e <= 0:
        raise ValueError(f"Oswald e must be > 0, got {e}")

    CDi = CL ** 2 / (np.pi * AR * e)

    # CDi must be non-negative (mathematically guaranteed, but guard floats)
    return max(float(CDi), 0.0)


def compute_total_drag(
    CL: float,
    AR: float,
    e: float,
    Re: float,
    S_ref: float,
    S_wet: float,
    form_factor: float = 1.2,
) -> DragBreakdown:
    """
    Compute total drag coefficient: CD = CD0 + CDi.

    Args:
        CL: Wing lift coefficient.
        AR: Aspect ratio.
        e: Oswald efficiency factor.
        Re: Reference Reynolds number (based on MAC).
        S_ref: Reference planform area (m^2).
        S_wet: Wetted surface area (m^2).
        form_factor: Airfoil form factor.

    Returns:
        DragBreakdown with CD0, CDi, and CD_total.
    """
    CD0 = compute_cd0(Re, S_ref, S_wet, form_factor)
    CDi = compute_cdi(CL, AR, e)
    CD_total = CD0 + CDi

    # Skin friction for diagnostics
    Re_safe = max(Re, 1e5)
    Cf = 0.455 / (np.log10(Re_safe)) ** 2.58

    return DragBreakdown(
        CD0=CD0,
        CDi=CDi,
        CD_total=CD_total,
        Cf=float(Cf),
        FF=form_factor,
        Re_ref=float(Re_safe),
    )


def compute_form_factor(
    thickness: float,
    thickness_loc: float,
) -> float:
    """
    Compute airfoil form factor for drag estimation.

    Method (Raymer):
        FF = 1 + 0.6/x_c * t/c + 100 * (t/c)^4

    where x_c is the chordwise location of max thickness and t/c is
    the max thickness ratio.

    Args:
        thickness: Max thickness ratio (t/c), e.g. 0.12.
        thickness_loc: Chordwise location of max thickness (x/c), e.g. 0.30.

    Returns:
        Form factor (typically 1.1-1.5 for subsonic airfoils).
    """
    x_c = max(thickness_loc, 0.05)  # Guard against division by zero
    FF = 1.0 + (0.6 / x_c) * thickness + 100.0 * thickness ** 4

    # Clamp to reasonable range
    return float(np.clip(FF, 1.0, 2.5))


def compute_wing_drag(
    CL: float,
    AR: float,
    e: float,
    rho: float,
    V: float,
    mu: float,
    root_chord: float,
    tip_chord: float,
    span: float,
    thickness: float,
    thickness_loc: float,
) -> DragBreakdown:
    """
    All-in-one wing drag computation from geometric inputs.

    Computes:
      - MAC (mean aerodynamic chord)
      - S_ref (reference planform area)
      - S_wet (wetted area, approx 2x planform with thickness correction)
      - Re_MAC
      - Form factor from thickness
      - CD0 + CDi = CD_total

    Args:
        CL: Wing lift coefficient.
        AR: Aspect ratio.
        e: Oswald efficiency factor.
        rho: Air density (kg/m^3).
        V: Freestream velocity (m/s).
        mu: Dynamic viscosity (Pa.s).
        root_chord, tip_chord: Chord lengths (m).
        span: Wingspan (m).
        thickness: Max t/c ratio.
        thickness_loc: Chordwise location of max thickness.

    Returns:
        DragBreakdown.
    """
    # Mean aerodynamic chord (trapezoidal approximation)
    c_mac = (root_chord + tip_chord) / 2.0

    # Reference area (trapezoidal planform)
    S_ref = c_mac * span

    # Wetted area (both surfaces, slightly larger for thick airfoils)
    S_wet = 2.0 * S_ref * (1.0 + 0.2 * thickness)

    # Reynolds number based on MAC
    Re_mac = rho * V * c_mac / mu

    # Form factor from airfoil geometry
    FF = compute_form_factor(thickness, thickness_loc)

    return compute_total_drag(CL, AR, e, Re_mac, S_ref, S_wet, FF)
