import numpy as np

from core.airfoil_ml import AirfoilML
from core.physics import enforce_physics, apply_stall
from core.validation import validate_wing_coefficients
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
    # Reynolds number (root approx)
    # ---------------------------
    chord_ref = inputs["chord_root"]
    Re = rho * V * chord_ref / mu
    Re = max(Re, 1e4)  # Numerical safety

    # ---------------------------
    # OOD check
    # ---------------------------
    x_full = np.concatenate([[alpha, Re], features])
    if not check_ood(x_full, mean, cov_inv):
        raise ValueError("Input airfoil outside training distribution")

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
    N = inputs.get("N", 20)
    theta = np.linspace(np.pi / (2 * N), np.pi / 2, N)
    y = (span / 2) * np.cos(theta)
    eta_s = np.cos(theta)

    # linear taper
    chord_root = inputs["chord_root"]
    chord_tip = inputs["chord_tip"]
    chord_dist = chord_root + (chord_tip - chord_root) * (1.0 - eta_s)

    # Reynolds variation
    Re_dist = rho * V * chord_dist / mu
    Re_dist = np.maximum(Re_dist, 1e4)

    # Sectional properties
    a0_dist = np.ones(N) * a0
    al0_dist = np.zeros(N)
    twist_deg = inputs.get("twist_deg", 0.0)
    tw_s_deg = twist_deg * (1.0 - eta_s)

    for i in range(N):
        # Local inputs
        alpha_i = alpha + tw_s_deg[i]
        
        # Predict with section-local Re, extracting unique Cd_i
        Cl_i, Cd_i = ml.predict(alpha_i, Re_dist[i], features)
        Cl_i, Cd_i = enforce_physics(alpha_i, Cl_i, Cd_i)
        Cl_i = apply_stall(Cl_i, alpha_i)
        
        # Compute effective zero-lift angle to force LLT consistent with ML Cl
        al0_dist[i] = np.radians(alpha_i) - (Cl_i / a0)

    # ---------------------------
    # LLT solve (Clean approach via Option A)
    # ---------------------------
    llt_res = solve_llt(
        N=N,
        alpha_wing_deg=alpha,
        span=span,
        root_chord=chord_root,
        tip_chord=chord_tip,
        twist_total_deg=twist_deg,
        a0_dist=a0_dist,
        al0_dist=al0_dist
    )

    if llt_res is None:
        raise RuntimeError("LLT numerical divergence")

    # ---------------------------
    # Wing-level coefficients
    # ---------------------------
    CL = llt_res.CL
    CDi = llt_res.CDi

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
    report = validate_wing_coefficients(CL, CD, CDi=CDi, e=llt_res.e, AR=AR)
    if not report.is_valid:
        raise RuntimeError("Invalid aerodynamic output")

    # ---------------------------
    # Output
    # ---------------------------
    return {
        "CL": CL,
        "CD": CD,
        "CD0": CD0,
        "CDi": CDi,
        "L_D": CL / CD,
        "spanwise_Cl": llt_res.cl_dist,
        "a0": a0,
        "Re": Re
    }
