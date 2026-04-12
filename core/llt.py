import numpy as np

def solve_llt(N, alpha_wing_deg, span, root_chord, tip_chord, twist_total_deg, a0_dist, al0_dist):
    """
    Prandtl's Lifting Line Theory (LLT) Solver.
    
    N: Number of stations
    alpha_wing_deg: Global wing angle of attack
    span: Total wingspan
    root_chord: Chord at the root
    tip_chord: Chord at the tip
    twist_total_deg: Total washout (negative for tip twist down)
    a0_dist: Distribution of section lift-curve slopes (per radian)
    al0_dist: Distribution of section zero-lift angles (radians)
    """
    theta = np.linspace(np.pi/(2*N), np.pi/2, N)
    y_s = (span/2) * np.cos(theta)
    
    # Linear taper assumption for chord distribution
    eta_s = y_s / (span/2)
    c_s = root_chord + (tip_chord - root_chord) * (1 - eta_s)
    
    # Linear twist distribution
    tw_s = twist_total_deg * (1 - eta_s)
    
    # Effective angle of attack at each station (radians)
    # alpha_local = alpha_wing + twist - alpha_zero_lift
    alpha_s = np.radians(alpha_wing_deg) + np.radians(tw_s) - al0_dist

    # Fourier Series Setup (Odd terms for symmetric lift)
    ns = np.arange(1, 2*N, 2)[:N]
    A_mat = np.zeros((N, N))
    rhs = np.zeros(N)
    
    for i in range(N):
        mu_i = c_s[i] * a0_dist[i] / (4 * span)
        for j, n in enumerate(ns):
            # Fourier matrix coefficients
            A_mat[i, j] = np.sin(n * theta[i]) * (1 + mu_i * n / np.sin(theta[i]))
        rhs[i] = mu_i * alpha_s[i]

    # Solve for An Fourier coefficients
    try:
        An = np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError:
        return None

    # Calculate global wing CL
    AR = (span**2) / (0.5 * (root_chord + tip_chord) * span)
    CL = An[0] * np.pi * AR
    
    # Calculate Induced Drag Coefficient (CDi)
    delta = np.sum(ns[1:] * (An[1:] / An[0])**2)
    e = 1.0 / (1.0 + delta)
    CDi = (CL**2) / (np.pi * AR * e)
    
    # Local lift coefficient distribution (Cl_dist)
    # Gamma = 2 * b * V * sum(An * sin(n * theta))
    # Cl = 2 * Gamma / (V * c) = 4 * b / c * sum(An * sin(n * theta))
    Cl_dist = np.zeros(N)
    for j, n in enumerate(ns):
        Cl_dist += An[j] * np.sin(n * theta)
    Cl_dist *= (4 * span / c_s)

    return {
        'CL': CL,
        'CDi': CDi,
        'e': e,
        'y_dist': y_s,
        'cl_dist': Cl_dist,
        'An': An
    }
