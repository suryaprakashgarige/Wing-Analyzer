import numpy as np

from core.airfoil_ml import AirfoilML
from core.physics import enforce_physics, apply_stall
from core.validation import validate_results
from core.llt import solve_llt
from core.drag import compute_cd0


# ---------------------------
# Utility: OOD check (Mahalanobis)
# ---------------------------
def check_ood(x, mean, cov_inv, threshold=3.0):
    diff = x - mean
    d = np.sqrt(diff.T @ cov_inv @ diff)
    return d < threshold


# ---------------------------
# Utility: robust lift slope
# ---------------------------
def compute_lift_slope(model, alpha, Re, features):
    delta = 2.0  # more stable than 1°

    alphas = np.array([alpha - delta, alpha, alpha + delta])
    cls = []

    for a in alphas:
        Cl, _ = model.predict(a, Re, features)
        cls.append(Cl)

    # quadratic fit for smoothing
    coeffs = np.polyfit(np.deg2rad(alphas), cls, 2)
    a0 = coeffs[1]

    # enforce physical bounds
    return np.clip(a0, 4.5, 7.0)


# ---------------------------
# Main Solver
# ---------------------------
def run_analysis(inputs):
    """
    inputs: dict containing:
        alpha (deg)
        velocity (m/s)
        rho (kg/m^3)
        mu (Pa.s)
        span (m)
        chord_root (m)
        chord_tip (m)
        area (m^2)
        AR
        e (initial guess)
        airfoil_features (list)
        ml_model_path
        training_mean
        training_cov_inv
        S_wet
        thickness_ratio
    """

    # ---------------------------
    # Load ML model
    # ---------------------------
    ml = AirfoilML(inputs["ml_model_path"])

    alpha = inputs["alpha"]
    V = inputs["velocity"]
    rho = inputs["rho"]
    mu = inputs["mu"]

    span = inputs["span"]
    S = inputs["area"]
    AR = inputs["AR"]

    features = np.array(inputs["airfoil_features"])
    mean = inputs["training_mean"]
    cov_inv = inputs["training_cov_inv"]

    # ---------------------------
    # OOD check
    # ---------------------------
    if not check_ood(features, mean, cov_inv):
        raise ValueError("Input airfoil outside training distribution")

    # ---------------------------
    # Reynolds number (root approx)
    # ---------------------------
    chord_ref = inputs["chord_root"]
    Re = rho * V * chord_ref / mu

    # ---------------------------
    # ML prediction (section)
    # ---------------------------
    Cl_ml, Cd_ml = ml.predict(alpha, Re, features)

    # ---------------------------
    # Physics enforcement
    # ---------------------------
    Cl, Cd = enforce_physics(alpha, Cl_ml, Cd_ml)
    Cl = apply_stall(Cl, alpha)

    # ---------------------------
    # Lift slope (critical for LLT)
    # ---------------------------
    a0 = compute_lift_slope(ml, alpha, Re, features)

    # ---------------------------
    # Spanwise discretization
    # ---------------------------
    N = 20
    y = np.linspace(-span/2, span/2, N)

    # linear taper
    chord_root = inputs["chord_root"]
    chord_tip = inputs["chord_tip"]

    chord_dist = chord_root + (chord_tip - chord_root) * (np.abs(y) / (span/2))

    # Reynolds variation
    Re_dist = rho * V * chord_dist / mu

    # Section Cl distribution (refined)
    Cl_dist = np.zeros(N)

    for i in range(N):
        Cl_i, _ = ml.predict(alpha, Re_dist[i], features)
        Cl_i, _ = enforce_physics(alpha, Cl_i, Cd_ml)
        Cl_i = apply_stall(Cl_i, alpha)
        Cl_dist[i] = Cl_i

    # Lift slope distribution
    a0_dist = np.ones(N) * a0

    # ---------------------------
    # LLT solve
    # ---------------------------
    An = solve_llt(
        N=N,
        alpha=alpha,
        chord=chord_dist,
        span=span,
        Cl_section=Cl_dist,
        a0_dist=a0_dist
    )

    # ---------------------------
    # Wing-level coefficients
    # ---------------------------
    CL = np.pi * AR * An[0]

    # induced drag
    e = inputs.get("e", 0.85)
    CDi = CL**2 / (np.pi * AR * e)

    # ---------------------------
    # Parasite drag
    # ---------------------------
    S_wet = inputs["S_wet"]
    t_c = inputs["thickness_ratio"]

    # improved form factor
    FF = 1 + 2.7 * t_c + 100 * t_c**4

    CD0_empirical = compute_cd0(Re, S, S_wet, FF)

    # blend with ML Cd (consistency)
    CD0 = 0.7 * CD0_empirical + 0.3 * max(Cd, 0.005)

    # total drag
    CD = CD0 + CDi

    # ---------------------------
    # Validation
    # ---------------------------
    validate_results(CL, CD)

    # ---------------------------
    # Output
    # ---------------------------
    return {
        "CL": CL,
        "CD": CD,
        "CD0": CD0,
        "CDi": CDi,
        "L_D": CL / CD,
        "spanwise_Cl": Cl_dist,
        "a0": a0,
        "Re": Re
    }
