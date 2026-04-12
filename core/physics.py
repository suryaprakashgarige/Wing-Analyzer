# core/physics.py
"""
Physics Constraint Module.

Enforces aerodynamic physical constraints on ML-predicted coefficients:
  1. Lift slope validation and blending
  2. Cl clamping to realistic bounds
  3. Cd minimum enforcement (skin friction floor)
  4. Stall model (post-stall Cl decay + Cd rise)
  5. Negative AoA symmetry handling

No UI code. No plotting. Pure physics enforcement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# --- Physical constants ---
TWO_PI = 2.0 * np.pi  # Thin-airfoil lift slope (per radian)

# --- Constraint bounds ---
CL_MAX = 2.0
CL_MIN = -1.5
CD_MIN_SKIN_FRICTION = 0.005  # Turbulent flat plate minimum
CD_MAX_REASONABLE = 0.30      # Beyond this, something is wrong

# --- Stall model defaults ---
DEFAULT_STALL_ALPHA = 15.0    # degrees, positive stall
DEFAULT_NEG_STALL_ALPHA = -12.0  # degrees, negative stall
STALL_DECAY_RATE = 5.0        # degrees, exponential decay width
POST_STALL_CD_RISE_RATE = 0.02  # Cd increase per degree past stall


@dataclass
class PhysicsResult:
    """Container for physics-constrained coefficients."""
    cl: float
    cd: float
    cl_unconstrained: float  # Pre-constraint value (for diagnostics)
    cd_unconstrained: float
    is_stalled: bool = False
    constraint_notes: list = None

    def __post_init__(self):
        if self.constraint_notes is None:
            self.constraint_notes = []


def compute_lift_slope_theoretical(thickness: float = 0.12) -> float:
    """
    Theoretical section lift-curve slope accounting for thickness.

    Thin-airfoil theory gives a0 = 2π per radian.
    Thickness correction (Abbott & Von Doenhoff):
        a0 = 2π * (1 + 0.77 * t/c)

    Args:
        thickness: Max thickness ratio (t/c).

    Returns:
        Corrected lift-curve slope (per radian).
    """
    return TWO_PI * (1.0 + 0.77 * thickness)


def enforce_physics(
    alpha_deg: float,
    cl_ml: float,
    cd_ml: float,
    thickness: float = 0.12,
    camber: float = 0.02,
    stall_alpha: float = DEFAULT_STALL_ALPHA,
) -> Tuple[float, float]:
    """
    Apply physics constraints to ML-predicted Cl, Cd.

    This is the main interface — returns only (Cl, Cd).

    Args:
        alpha_deg: Angle of attack in degrees.
        cl_ml: ML-predicted lift coefficient.
        cd_ml: ML-predicted drag coefficient.
        thickness: Airfoil max thickness ratio.
        camber: Airfoil max camber ratio.
        stall_alpha: Stall angle of attack in degrees.

    Returns:
        Tuple of (Cl_constrained, Cd_constrained).
    """
    result = enforce_physics_detailed(
        alpha_deg, cl_ml, cd_ml, thickness, camber, stall_alpha
    )
    return result.cl, result.cd


def enforce_physics_detailed(
    alpha_deg: float,
    cl_ml: float,
    cd_ml: float,
    thickness: float = 0.12,
    camber: float = 0.02,
    stall_alpha: float = DEFAULT_STALL_ALPHA,
) -> PhysicsResult:
    """
    Full physics enforcement with diagnostic metadata.

    Strategy:
      1. In the linear region (|alpha| < stall_alpha):
         - Compute theoretical Cl from thin-airfoil + thickness correction
         - BLEND between ML prediction and theoretical (70% ML, 30% theory)
         - This trusts the ML but pulls it toward physical reality
      2. In the post-stall region:
         - Apply exponential decay to Cl
         - Apply Cd rise proportional to (alpha - alpha_stall)
      3. Always enforce minimum Cd (skin friction floor)
      4. Always clamp Cl to physical bounds
    """
    alpha_rad = np.deg2rad(alpha_deg)
    notes = []
    is_stalled = False

    cl_unconstrained = cl_ml
    cd_unconstrained = cd_ml

    # --- Step 1: Lift slope blending in linear region ---
    # Zero-lift angle estimate from camber (thin-airfoil: alpha_L0 ≈ -2 * camber for typical NACA)
    alpha_l0_rad = -2.0 * camber  # radians, rough estimate
    a0_theory = compute_lift_slope_theoretical(thickness)
    cl_theory = a0_theory * (alpha_rad - alpha_l0_rad)

    if abs(alpha_deg) < stall_alpha * 0.9:  # Use 90% of stall angle as "safe linear"
        # Blend: trust ML more than theory, but anchor to physics
        blend_weight = 0.70  # 70% ML, 30% theory
        cl = blend_weight * cl_ml + (1.0 - blend_weight) * cl_theory

        # Check if ML and theory disagree significantly
        deviation = abs(cl_ml - cl_theory)
        if deviation > 0.5:
            notes.append(
                f"ML/theory Cl deviation={deviation:.3f} at alpha={alpha_deg:.1f}°"
            )
    else:
        # Near or past stall — rely more on ML with stall model applied
        cl = cl_ml

    # --- Step 2: Post-stall model ---
    cl, is_stalled_pos = _apply_stall(cl, alpha_deg, stall_alpha)
    cl, is_stalled_neg = _apply_stall_negative(cl, alpha_deg)
    is_stalled = is_stalled_pos or is_stalled_neg

    if is_stalled:
        notes.append(f"Stall applied at alpha={alpha_deg:.1f}°")

    # --- Step 3: Cl clamping ---
    if cl > CL_MAX:
        notes.append(f"Cl clamped from {cl:.3f} to {CL_MAX}")
        cl = CL_MAX
    elif cl < CL_MIN:
        notes.append(f"Cl clamped from {cl:.3f} to {CL_MIN}")
        cl = CL_MIN

    # --- Step 4: Cd enforcement ---
    cd = cd_ml

    # Post-stall drag rise
    if is_stalled:
        if alpha_deg > stall_alpha:
            cd += POST_STALL_CD_RISE_RATE * (alpha_deg - stall_alpha)
        elif alpha_deg < DEFAULT_NEG_STALL_ALPHA:
            cd += POST_STALL_CD_RISE_RATE * abs(alpha_deg - DEFAULT_NEG_STALL_ALPHA)

    # Minimum drag floor
    if cd < CD_MIN_SKIN_FRICTION:
        notes.append(f"Cd floored from {cd:.6f} to {CD_MIN_SKIN_FRICTION}")
        cd = CD_MIN_SKIN_FRICTION

    # Maximum drag sanity
    if cd > CD_MAX_REASONABLE:
        notes.append(f"Cd capped from {cd:.4f} to {CD_MAX_REASONABLE}")
        cd = CD_MAX_REASONABLE

    return PhysicsResult(
        cl=cl,
        cd=cd,
        cl_unconstrained=cl_unconstrained,
        cd_unconstrained=cd_unconstrained,
        is_stalled=is_stalled,
        constraint_notes=notes,
    )


def _apply_stall(cl: float, alpha_deg: float, stall_alpha: float) -> Tuple[float, bool]:
    """
    Post-stall Cl decay for positive angles.

    Uses Kirchhoff/Helmholtz-inspired trailing-edge separation model:
        Cl_post = Cl_max * ((1 + sqrt(f))/2)^2
    where f decays exponentially past stall.

    Simplified here as exponential decay for robustness.
    """
    if alpha_deg <= stall_alpha:
        return cl, False

    excess = alpha_deg - stall_alpha
    decay = np.exp(-excess / STALL_DECAY_RATE)
    return cl * decay, True


def _apply_stall_negative(cl: float, alpha_deg: float) -> Tuple[float, bool]:
    """
    Post-stall Cl decay for negative angles (inverted flight, negative camber effects).
    """
    if alpha_deg >= DEFAULT_NEG_STALL_ALPHA:
        return cl, False

    excess = abs(alpha_deg - DEFAULT_NEG_STALL_ALPHA)
    decay = np.exp(-excess / STALL_DECAY_RATE)
    return cl * decay, True


def enforce_physics_batch(
    alpha_deg_array: np.ndarray,
    cl_array: np.ndarray,
    cd_array: np.ndarray,
    thickness: float = 0.12,
    camber: float = 0.02,
    stall_alpha: float = DEFAULT_STALL_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized physics enforcement for spanwise distributions.

    Args:
        alpha_deg_array: Array of AoA in degrees.
        cl_array: ML-predicted Cl values.
        cd_array: ML-predicted Cd values.
        thickness: Airfoil thickness ratio.
        camber: Airfoil camber ratio.
        stall_alpha: Stall angle.

    Returns:
        Tuple of (cl_constrained, cd_constrained) arrays.
    """
    n = len(alpha_deg_array)
    cl_out = np.empty(n)
    cd_out = np.empty(n)

    for i in range(n):
        cl_out[i], cd_out[i] = enforce_physics(
            alpha_deg_array[i], cl_array[i], cd_array[i],
            thickness, camber, stall_alpha,
        )

    return cl_out, cd_out
