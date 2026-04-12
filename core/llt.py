# core/llt.py
"""
Prandtl Lifting Line Theory (LLT) Solver.

Solves for the spanwise circulation distribution on a finite wing
using Fourier series decomposition. Returns wing-level aerodynamic
coefficients (CL, CDi, Oswald factor) and spanwise distributions.

Implements:
  - Glauert's method (Fourier decomposition of circulation)
  - Matrix conditioning checks for numerical stability
  - Induced angle of attack distribution
  - Span efficiency (Oswald) factor

No UI code. No plotting. Pure numerical solver.

References:
  - Anderson, J.D. "Fundamentals of Aerodynamics", Ch. 5
  - Phillips, W.F. "Mechanics of Flight", Ch. 1.8
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# --- Numerical safety thresholds ---
CONDITION_NUMBER_WARNING = 1e8
CONDITION_NUMBER_REJECT = 1e12
MIN_STATIONS = 8
MAX_STATIONS = 200
MIN_SPAN = 0.1  # meters
MIN_CHORD = 0.01  # meters


@dataclass
class LLTResult:
    """Container for LLT solution."""
    CL: float                     # Wing lift coefficient
    CDi: float                    # Induced drag coefficient
    e: float                      # Oswald span efficiency factor
    AR: float                     # Aspect ratio
    An: np.ndarray                # Fourier coefficients
    y_dist: np.ndarray            # Spanwise station positions (m)
    cl_dist: np.ndarray           # Local lift coefficient distribution
    alpha_i_dist: np.ndarray      # Induced AoA distribution (radians)
    gamma_dist: np.ndarray        # Circulation distribution (normalized)
    cond_number: float            # Matrix condition number
    warnings: list = field(default_factory=list)


class LLTError(Exception):
    """Raised when LLT solver encounters an unrecoverable error."""
    pass


def solve_llt(
    N: int,
    alpha_wing_deg: float,
    span: float,
    root_chord: float,
    tip_chord: float,
    twist_total_deg: float,
    a0_dist: np.ndarray,
    al0_dist: np.ndarray,
) -> Optional[LLTResult]:
    """
    Solve Prandtl's Lifting Line equation using Glauert's Fourier method.

    Formulation (Anderson, Fundamentals of Aerodynamics, 5th Ed., Sec 5.3):

    The fundamental equation of Prandtl's LLT:
        α(y) = Γ(y)/(π V c(y)) + α_L0(y) + α_i(y)

    where Γ(y) is expanded as:
        Γ(θ) = 2bV Σ An sin(nθ)

    Leading to the system:
        Σ An sin(nθ_i)[1 + (n μ_i)/sin(θ_i)] = μ_i α_eff(θ_i)

    where μ_i = c(θ_i) a0(θ_i) / (4b)

    Args:
        N: Number of spanwise stations (8-200, odd recommended).
        alpha_wing_deg: Global wing angle of attack (degrees).
        span: Total wingspan (m).
        root_chord: Root chord length (m).
        tip_chord: Tip chord length (m).
        twist_total_deg: Total geometric twist from root to tip (degrees).
                         Negative = washout (tip nose-down).
        a0_dist: Array [N] of section lift-curve slopes (per radian).
        al0_dist: Array [N] of section zero-lift angles (radians).

    Returns:
        LLTResult on success, None only if matrix is completely singular.

    Raises:
        LLTError: On invalid inputs.
    """
    # --- Input validation ---
    _validate_inputs(N, span, root_chord, tip_chord, a0_dist, al0_dist)

    warnings = []
    b = span / 2.0  # semi-span

    # --- Spanwise discretization ---
    # Use cosine-spaced stations (excludes root θ=0 and tip θ=π/2 singularities)
    theta = np.linspace(np.pi / (2 * N), np.pi / 2, N)
    y_s = b * np.cos(theta)  # Physical spanwise positions

    # --- Chord distribution (linear taper) ---
    eta_s = y_s / b  # Normalized span position (0=tip, 1=root)
    c_s = root_chord + (tip_chord - root_chord) * (1.0 - eta_s)

    # Validate chords
    if np.any(c_s <= 0):
        raise LLTError(
            f"Negative/zero chord detected. root={root_chord}, tip={tip_chord}. "
            f"Chord range: [{c_s.min():.4f}, {c_s.max():.4f}]"
        )

    # --- Twist distribution (linear) ---
    tw_s_deg = twist_total_deg * (1.0 - eta_s)  # Zero at root, full at tip

    # --- Effective angle of attack at each station ---
    alpha_eff = np.radians(alpha_wing_deg) + np.radians(tw_s_deg) - al0_dist

    # --- Build Fourier system ---
    # Use odd harmonics only for symmetric lift distribution
    ns = np.arange(1, 2 * N, 2)[:N]  # [1, 3, 5, ..., 2N-1]

    A_mat = np.zeros((N, N))
    rhs = np.zeros(N)

    for i in range(N):
        mu_i = c_s[i] * a0_dist[i] / (4.0 * span)
        sin_theta_i = np.sin(theta[i])

        # Guard against sin(θ) → 0 (shouldn't happen with proper spacing)
        if abs(sin_theta_i) < 1e-12:
            warnings.append(f"Near-zero sin(theta) at station {i}, theta={theta[i]:.6f}")
            sin_theta_i = 1e-12

        for j, n in enumerate(ns):
            A_mat[i, j] = np.sin(n * theta[i]) * (1.0 + mu_i * n / sin_theta_i)

        rhs[i] = mu_i * alpha_eff[i]

    # --- Condition number check ---
    cond = np.linalg.cond(A_mat)

    if cond > CONDITION_NUMBER_REJECT:
        warnings.append(f"Matrix condition number {cond:.2e} exceeds reject threshold")
        return None

    if cond > CONDITION_NUMBER_WARNING:
        warnings.append(
            f"Matrix condition number {cond:.2e} is high — results may be inaccurate"
        )

    # --- Solve ---
    try:
        An = np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError as e:
        warnings.append(f"LinAlgError: {e}")
        # Fall back to least-squares
        try:
            An, residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            warnings.append(f"Used lstsq fallback, rank={rank}")
        except Exception:
            return None

    # --- Sanity check on A1 ---
    if abs(An[0]) < 1e-15:
        warnings.append("A1 ≈ 0 — wing produces essentially no lift")

    # --- Wing Aspect Ratio ---
    S_wing = 0.5 * (root_chord + tip_chord) * span  # Trapezoidal approximation
    AR = span**2 / S_wing

    # --- Wing CL ---
    CL = An[0] * np.pi * AR

    # --- Induced Drag CDi ---
    if abs(An[0]) > 1e-15:
        delta = np.sum(ns[1:] * (An[1:] / An[0]) ** 2)
    else:
        delta = 0.0

    e = 1.0 / (1.0 + delta) if (1.0 + delta) > 0 else 0.0

    # Validate Oswald factor
    if not (0.0 < e <= 1.0):
        warnings.append(f"Oswald factor e={e:.4f} outside (0, 1] — clamping")
        e = np.clip(e, 0.01, 1.0)

    if AR > 0 and e > 0:
        CDi = CL**2 / (np.pi * AR * e)
    else:
        CDi = 0.0
        warnings.append("CDi set to 0 due to invalid AR or e")

    # Ensure CDi is non-negative
    CDi = max(CDi, 0.0)

    # --- Spanwise distributions ---
    # Local circulation: Γ(θ) = 2bV Σ An sin(nθ) → normalized = Σ An sin(nθ)
    gamma_dist = np.zeros(N)
    for j, n in enumerate(ns):
        gamma_dist += An[j] * np.sin(n * theta)

    # Local lift coefficient: Cl(y) = 2Γ/(Vc) = 4b/c * Σ An sin(nθ)
    cl_dist = (4.0 * span / c_s) * gamma_dist

    # Induced angle of attack: α_i = Σ (n An sin(nθ)) / sin(θ)
    alpha_i_dist = np.zeros(N)
    for j, n in enumerate(ns):
        alpha_i_dist += n * An[j] * np.sin(n * theta)
    sin_theta_safe = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
    alpha_i_dist /= sin_theta_safe

    return LLTResult(
        CL=CL,
        CDi=CDi,
        e=e,
        AR=AR,
        An=An,
        y_dist=y_s,
        cl_dist=cl_dist,
        alpha_i_dist=alpha_i_dist,
        gamma_dist=gamma_dist,
        cond_number=cond,
        warnings=warnings,
    )


def _validate_inputs(
    N: int,
    span: float,
    root_chord: float,
    tip_chord: float,
    a0_dist: np.ndarray,
    al0_dist: np.ndarray,
) -> None:
    """Validate LLT solver inputs."""
    if not (MIN_STATIONS <= N <= MAX_STATIONS):
        raise LLTError(f"N={N} must be in [{MIN_STATIONS}, {MAX_STATIONS}]")

    if span < MIN_SPAN:
        raise LLTError(f"Span={span} must be >= {MIN_SPAN}")

    if root_chord < MIN_CHORD:
        raise LLTError(f"Root chord={root_chord} must be >= {MIN_CHORD}")

    if tip_chord < MIN_CHORD:
        raise LLTError(f"Tip chord={tip_chord} must be >= {MIN_CHORD}")

    if len(a0_dist) != N:
        raise LLTError(f"a0_dist length ({len(a0_dist)}) != N ({N})")

    if len(al0_dist) != N:
        raise LLTError(f"al0_dist length ({len(al0_dist)}) != N ({N})")

    if np.any(np.isnan(a0_dist)) or np.any(np.isnan(al0_dist)):
        raise LLTError("NaN detected in section property distributions")

    # Sanity: lift-curve slope should be positive and near 2π
    if np.any(a0_dist <= 0):
        raise LLTError(
            f"Non-positive lift-curve slopes detected. "
            f"Range: [{a0_dist.min():.2f}, {a0_dist.max():.2f}]"
        )
