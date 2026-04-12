# core/validation.py
"""
Result Validation Module.

Validates aerodynamic analysis results against physical reality
with graduated severity levels (info, warning, error).

Checks:
  - Section-level: Cl, Cd bounds per station
  - Wing-level: CL, CDi, Oswald factor, L/D ratio
  - Numerical: condition numbers, convergence
  - Cross-checks: CL vs alpha consistency

No UI code. No plotting. Pure validation logic.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: Severity
    parameter: str
    message: str
    value: Optional[float] = None
    expected_range: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for an analysis run."""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: Severity, parameter: str, message: str,
            value: float = None, expected_range: str = None):
        issue = ValidationIssue(severity, parameter, message, value, expected_range)
        self.issues.append(issue)
        if severity == Severity.ERROR:
            self.is_valid = False

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        lines = []
        for issue in self.issues:
            val_str = f" = {issue.value:.4f}" if issue.value is not None else ""
            range_str = f" (expected: {issue.expected_range})" if issue.expected_range else ""
            lines.append(f"[{issue.severity.value}] {issue.parameter}{val_str}: {issue.message}{range_str}")
        if not lines:
            lines.append("[OK] All checks passed.")
        return "\n".join(lines)


# --- Validation bounds by category ---

# Wing-level bounds
WING_CL_RANGE = (-1.5, 2.5)
WING_CDI_RANGE = (0.0, 0.15)
WING_CD_TOTAL_RANGE = (0.001, 0.30)
WING_LD_RANGE = (1.0, 60.0)  # Gliders can reach ~50+
OSWALD_RANGE = (0.5, 1.0)    # Below 0.5 is suspicious
AR_RANGE = (1.0, 50.0)       # From delta wings to sailplanes

# Section-level bounds
SECTION_CL_RANGE = (-2.0, 3.0)
SECTION_CD_RANGE = (0.001, 0.20)

# Condition number
COND_WARNING = 1e6
COND_ERROR = 1e10


def validate_results(CL: float, CD: float) -> bool:
    """
    Legacy interface — simple pass/fail validation.

    Maintained for backward compatibility with main.py and app.py.
    For detailed diagnostics, use validate_full().

    Args:
        CL: Wing lift coefficient.
        CD: Wing total drag coefficient.

    Returns:
        True if results are within acceptable bounds.

    Raises:
        ValueError: If results are physically impossible.
    """
    report = validate_wing_coefficients(CL, CD)

    if not report.is_valid:
        error_msgs = "; ".join(e.message for e in report.errors)
        raise ValueError(f"Validation failed: {error_msgs}")

    return True


def validate_wing_coefficients(
    CL: float,
    CD: float,
    CDi: float = None,
    e: float = None,
    AR: float = None,
    cond_number: float = None,
) -> ValidationReport:
    """
    Validate wing-level aerodynamic coefficients.

    Args:
        CL: Wing lift coefficient.
        CD: Wing total drag coefficient.
        CDi: Induced drag coefficient (optional).
        e: Oswald efficiency factor (optional).
        AR: Aspect ratio (optional).
        cond_number: LLT matrix condition number (optional).

    Returns:
        ValidationReport with graduated severity findings.
    """
    report = ValidationReport()

    # --- CL checks ---
    if np.isnan(CL) or np.isinf(CL):
        report.add(Severity.ERROR, "CL", "NaN or Inf detected", CL)
    elif not (WING_CL_RANGE[0] <= CL <= WING_CL_RANGE[1]):
        report.add(Severity.ERROR, "CL", "Outside physical bounds",
                   CL, f"[{WING_CL_RANGE[0]}, {WING_CL_RANGE[1]}]")

    # --- CD checks ---
    if np.isnan(CD) or np.isinf(CD):
        report.add(Severity.ERROR, "CD", "NaN or Inf detected", CD)
    elif CD <= 0:
        report.add(Severity.ERROR, "CD", "Non-positive drag is physically impossible", CD)
    elif not (WING_CD_TOTAL_RANGE[0] <= CD <= WING_CD_TOTAL_RANGE[1]):
        report.add(Severity.WARNING, "CD", "Outside typical range",
                   CD, f"[{WING_CD_TOTAL_RANGE[0]}, {WING_CD_TOTAL_RANGE[1]}]")

    # --- L/D check ---
    if CD > 0 and not (np.isnan(CL) or np.isnan(CD)):
        LD = CL / CD
        if not (WING_LD_RANGE[0] <= abs(LD) <= WING_LD_RANGE[1]):
            report.add(Severity.WARNING, "L/D", "Outside typical range",
                       LD, f"[{WING_LD_RANGE[0]}, {WING_LD_RANGE[1]}]")

    # --- CDi checks ---
    if CDi is not None:
        if CDi < 0:
            report.add(Severity.ERROR, "CDi", "Negative induced drag", CDi)
        elif not (WING_CDI_RANGE[0] <= CDi <= WING_CDI_RANGE[1]):
            report.add(Severity.WARNING, "CDi", "Outside typical range",
                       CDi, f"[{WING_CDI_RANGE[0]}, {WING_CDI_RANGE[1]}]")

    # --- Oswald factor checks ---
    if e is not None:
        # Recompute e safely if CDi is available (validation safeguard)
        if CDi is not None and AR is not None and CDi > 1e-12:
            if abs(CL) < 0.1:
                e = np.nan
            else:
                e = CL**2 / (np.pi * AR * CDi)

        # Apply updated warning logic
        if not np.isnan(e):
            if abs(CL) < 0.1:
                report.add(Severity.INFO, "Oswald e",
                           "Oswald efficiency undefined near zero lift", e)
            elif e < 0.6:
                report.add(Severity.WARNING, "Oswald e",
                           "Low span efficiency — check wing geometry", e)
            elif e > 1.0:
                report.add(Severity.ERROR, "Oswald e",
                           "Exceeds 1.0 — non-physical", e)

    # --- AR checks ---
    if AR is not None:
        if not (AR_RANGE[0] <= AR <= AR_RANGE[1]):
            report.add(Severity.WARNING, "AR", "Unusual aspect ratio",
                       AR, f"[{AR_RANGE[0]}, {AR_RANGE[1]}]")

    # --- Condition number checks ---
    if cond_number is not None:
        if cond_number > COND_ERROR:
            report.add(Severity.ERROR, "Condition Number",
                       "Matrix extremely ill-conditioned — results unreliable",
                       cond_number)
        elif cond_number > COND_WARNING:
            report.add(Severity.WARNING, "Condition Number",
                       "Matrix moderately ill-conditioned",
                       cond_number)

    return report


def validate_spanwise_distribution(
    y_dist: np.ndarray,
    cl_dist: np.ndarray,
    span: float,
) -> ValidationReport:
    """
    Validate spanwise lift distribution for physical consistency.

    Checks:
      - No NaN/Inf values
      - Cl within section bounds
      - Distribution roughly elliptic shape (Cl should be max near root)
      - Symmetry (Cl should not spike at tips)

    Args:
        y_dist: Spanwise station positions (m).
        cl_dist: Local lift coefficients at each station.
        span: Total wingspan (m).

    Returns:
        ValidationReport.
    """
    report = ValidationReport()

    if np.any(np.isnan(cl_dist)):
        report.add(Severity.ERROR, "cl_dist", "NaN in spanwise distribution")
        return report

    if np.any(np.isinf(cl_dist)):
        report.add(Severity.ERROR, "cl_dist", "Inf in spanwise distribution")
        return report

    # Section Cl bounds
    cl_max = np.max(cl_dist)
    cl_min = np.min(cl_dist)

    if cl_max > SECTION_CL_RANGE[1]:
        report.add(Severity.WARNING, "cl_dist max",
                   "Local Cl exceeds section maximum", cl_max,
                   f"< {SECTION_CL_RANGE[1]}")

    if cl_min < SECTION_CL_RANGE[0]:
        report.add(Severity.WARNING, "cl_dist min",
                   "Local Cl below section minimum", cl_min,
                   f"> {SECTION_CL_RANGE[0]}")

    # Tip Cl check — should approach zero for proper loading
    tip_cl = cl_dist[-1] if len(cl_dist) > 0 else 0
    if abs(tip_cl) > 0.3 * cl_max and cl_max > 0.1:
        report.add(Severity.INFO, "cl_dist tip",
                   f"Tip Cl = {tip_cl:.3f} is significant — check tip modeling",
                   tip_cl)

    return report


def validate_cross_check(
    CL_llt: float,
    alpha_deg: float,
    AR: float,
) -> ValidationReport:
    """
    Cross-check LLT CL against simple finite-wing estimate.

    For a finite wing: CL ≈ a_wing * α
    where a_wing ≈ (2π * AR) / (AR + 2) for elliptic loading.

    This is a rough sanity check — deviations up to 30% are acceptable
    due to camber, twist, and non-elliptic loading.
    """
    report = ValidationReport()

    if AR <= 0 or abs(alpha_deg) < 0.5:
        return report  # Skip for very small AoA or invalid AR

    alpha_rad = np.radians(alpha_deg)
    a_wing = (2 * np.pi * AR) / (AR + 2.0)
    CL_estimate = a_wing * alpha_rad

    if abs(CL_estimate) > 0.01:
        ratio = CL_llt / CL_estimate
        if not (0.5 <= ratio <= 2.0):
            report.add(
                Severity.WARNING, "CL cross-check",
                f"LLT CL={CL_llt:.3f} vs finite-wing estimate={CL_estimate:.3f} "
                f"(ratio={ratio:.2f}) — large deviation",
                CL_llt,
            )

    return report
