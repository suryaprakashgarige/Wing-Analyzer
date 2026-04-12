# core/ood.py
"""
Out-of-Distribution (OOD) Detection Module.

Detects when ML model inputs fall outside the training data distribution
using z-score distance. When inputs are OOD, the prediction should be
rejected or replaced with a physics-based fallback.

No UI code. No plotting. Pure statistical logic.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OODResult:
    """Result of an out-of-distribution check."""
    is_in_distribution: bool
    z_scores: np.ndarray       # Per-feature z-scores
    max_z: float               # Maximum z-score across features
    flagged_features: list     # Names of features that exceeded threshold
    threshold: float           # Threshold used


# --- Training data statistics ---
# These represent the mean and std of each feature in the training dataset.
# Feature order: [Re, alpha_deg, thickness, thickness_loc, camber, camber_loc]
#
# These MUST be updated if the model is retrained on different data.
# Current values are estimated from typical XFOIL/airfoil datasets.

FEATURE_NAMES = ['Re', 'alpha_deg', 'thickness', 'thickness_loc', 'camber', 'camber_loc']

# Training distribution statistics (mean, std)
# Conservative estimates covering common airfoil analysis datasets
TRAINING_MEAN = np.array([
    3e6,      # Re: typical range 1e5 to 1e7
    5.0,      # alpha: typical range -5 to 15 deg
    0.12,     # thickness: typical NACA range 0.06 to 0.24
    0.30,     # thickness_loc: typical 0.20 to 0.40
    0.02,     # camber: typical 0 to 0.06
    0.40,     # camber_loc: typical 0.20 to 0.50
])

TRAINING_STD = np.array([
    2e6,      # Re
    5.0,      # alpha
    0.04,     # thickness
    0.05,     # thickness_loc
    0.015,    # camber
    0.08,     # camber_loc
])


def check_out_of_distribution(
    x: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    threshold: float = 3.0,
) -> OODResult:
    """
    Check if input feature vector is within the training distribution.

    Uses z-score distance: z = |x - mean| / std
    If any feature has z > threshold, the input is flagged as OOD.

    Args:
        x: Input feature vector (1D array of length 6).
            Order: [Re, alpha_deg, thickness, thickness_loc, camber, camber_loc]
        mean: Training data mean per feature. If None, uses default.
        std: Training data std per feature. If None, uses default.
        threshold: Z-score threshold (default 3.0 = 99.7% of training data).

    Returns:
        OODResult with per-feature z-scores and overall verdict.
    """
    if mean is None:
        mean = TRAINING_MEAN
    if std is None:
        std = TRAINING_STD

    x = np.asarray(x, dtype=float)

    if x.shape != mean.shape:
        raise ValueError(
            f"Feature vector shape {x.shape} != expected {mean.shape}"
        )

    # Guard against zero std (constant feature)
    std_safe = np.where(std > 0, std, 1e-12)

    # Compute z-scores
    z_scores = np.abs((x - mean) / std_safe)
    max_z = float(np.max(z_scores))

    # Flag features exceeding threshold
    flagged = []
    for i, (z, name) in enumerate(zip(z_scores, FEATURE_NAMES)):
        if z > threshold:
            flagged.append(f"{name} (z={z:.2f})")

    is_in_dist = len(flagged) == 0

    return OODResult(
        is_in_distribution=is_in_dist,
        z_scores=z_scores,
        max_z=max_z,
        flagged_features=flagged,
        threshold=threshold,
    )


def check_prediction_ood(
    re: float,
    alpha_deg: float,
    thickness: float,
    thickness_loc: float,
    camber: float,
    camber_loc: float,
    threshold: float = 3.0,
) -> OODResult:
    """
    Convenience wrapper — takes individual prediction inputs.

    Args:
        re: Reynolds number.
        alpha_deg: Angle of attack in degrees.
        thickness: Max thickness ratio.
        thickness_loc: Chordwise location of max thickness.
        camber: Max camber ratio.
        camber_loc: Chordwise location of max camber.
        threshold: Z-score threshold (default 3.0).

    Returns:
        OODResult.
    """
    x = np.array([re, alpha_deg, thickness, thickness_loc, camber, camber_loc])
    return check_out_of_distribution(x, threshold=threshold)
