import numpy as np

def enforce_physics(alpha, Cl, Cd):
    """
    Enforces aerodynamic constraints on predicted coefficients.
    1. Linear lift slope correction.
    2. Clamping Cl to realistic bounds.
    3. Ensuring Cd is not below minimum skin friction.
    """
    # Convert alpha to radians
    alpha_rad = np.deg2rad(alpha)

    # Lift slope correction (linear region)
    if abs(alpha) < 10:
        Cl = 2 * np.pi * alpha_rad

    # Clamp lift
    Cl = np.clip(Cl, -1.5, 2.0)

    # Ensure Cd is valid (minimum drag for a smooth plate is approx 0.005)
    Cd = max(Cd, 0.005)

    return Cl, Cd


def apply_stall(Cl, alpha):
    """
    Applies empirical stall behavior for high angles of attack.
    """
    alpha_stall = 15

    if alpha > alpha_stall:
        Cl *= np.exp(-(alpha - alpha_stall)/5)

    return Cl
