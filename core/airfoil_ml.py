# core/airfoil_ml.py
"""
Airfoil ML Prediction Module.

Loads trained scikit-learn models (joblib) and returns Cl, Cd predictions
with input validation and output sanity checking.

No UI code. No plotting. Pure prediction logic.
"""

import joblib
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple, Optional


# --- Physical bounds for sanity checking ---
_RE_MIN = 1e4
_RE_MAX = 1e8
_AOA_MIN = -20.0  # degrees
_AOA_MAX = 25.0   # degrees
_THICKNESS_MIN = 0.01
_THICKNESS_MAX = 0.30
_CAMBER_MIN = 0.0
_CAMBER_MAX = 0.10

# Output sanity bounds (raw ML output — physics module clamps further)
_CL_RAW_MIN = -3.0
_CL_RAW_MAX = 3.0
_CD_RAW_MIN = -0.01  # ML can produce slightly negative — we flag it
_CD_RAW_MAX = 0.50


@dataclass
class AirfoilPrediction:
    """Container for a single airfoil prediction with metadata."""
    cl: float
    cd: float
    is_extrapolated: bool = False  # True if inputs were outside training envelope
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AirfoilML:
    """
    ML-based airfoil coefficient predictor.

    Expects two joblib-serialized scikit-learn models:
      - model_cl.pkl: predicts lift coefficient
      - model_cd.pkl: predicts drag coefficient

    Feature vector order: [Re, alpha, thickness, thickness_loc, camber, camber_loc]
    """

    def __init__(self, models_dir: str):
        """
        Load trained models from disk.

        Args:
            models_dir: Directory containing model_cl.pkl and model_cd.pkl.

        Raises:
            FileNotFoundError: If either model file is missing.
            RuntimeError: If models fail to load.
        """
        cl_path = os.path.join(models_dir, 'model_cl.pkl')
        cd_path = os.path.join(models_dir, 'model_cd.pkl')

        if not os.path.isfile(cl_path):
            raise FileNotFoundError(f"CL model not found: {cl_path}")
        if not os.path.isfile(cd_path):
            raise FileNotFoundError(f"CD model not found: {cd_path}")

        try:
            self.model_cl = joblib.load(cl_path)
            self.model_cd = joblib.load(cd_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load models from {models_dir}: {e}")

        self._models_dir = models_dir
        self._n_features = 6  # Expected feature count

    def predict(
        self,
        re: float,
        aoa: float,
        thickness: float,
        thickness_loc: float,
        camber: float,
        camber_loc: float,
    ) -> Tuple[float, float]:
        """
        Predict Cl and Cd for a single operating condition.

        Args:
            re: Reynolds number.
            aoa: Angle of attack in degrees.
            thickness: Maximum thickness ratio (e.g. 0.12).
            thickness_loc: Chordwise location of max thickness (0-1).
            camber: Maximum camber ratio (e.g. 0.02).
            camber_loc: Chordwise location of max camber (0-1).

        Returns:
            Tuple of (Cl, Cd) with basic sanity applied.

        Raises:
            ValueError: If inputs contain NaN or Inf.
        """
        prediction = self.predict_detailed(
            re, aoa, thickness, thickness_loc, camber, camber_loc
        )
        return prediction.cl, prediction.cd

    def predict_detailed(
        self,
        re: float,
        aoa: float,
        thickness: float,
        thickness_loc: float,
        camber: float,
        camber_loc: float,
    ) -> AirfoilPrediction:
        """
        Predict with full metadata (extrapolation flags, warnings).
        """
        # --- Input validation ---
        inputs = [re, aoa, thickness, thickness_loc, camber, camber_loc]
        if any(np.isnan(v) or np.isinf(v) for v in inputs):
            raise ValueError(
                f"NaN or Inf in inputs: Re={re}, aoa={aoa}, "
                f"t={thickness}, t_loc={thickness_loc}, "
                f"c={camber}, c_loc={camber_loc}"
            )

        warnings = []
        is_extrapolated = False

        # Check input ranges
        if not (_RE_MIN <= re <= _RE_MAX):
            warnings.append(f"Re={re:.0f} outside training range [{_RE_MIN:.0e}, {_RE_MAX:.0e}]")
            is_extrapolated = True

        if not (_AOA_MIN <= aoa <= _AOA_MAX):
            warnings.append(f"AoA={aoa:.1f}° outside training range [{_AOA_MIN}, {_AOA_MAX}]")
            is_extrapolated = True

        if not (_THICKNESS_MIN <= thickness <= _THICKNESS_MAX):
            warnings.append(f"Thickness={thickness:.3f} outside range [{_THICKNESS_MIN}, {_THICKNESS_MAX}]")
            is_extrapolated = True

        if not (_CAMBER_MIN <= camber <= _CAMBER_MAX):
            warnings.append(f"Camber={camber:.4f} outside range [{_CAMBER_MIN}, {_CAMBER_MAX}]")
            is_extrapolated = True

        # --- Build feature vector ---
        x = np.array([[re, aoa, thickness, thickness_loc, camber, camber_loc]])

        # --- Predict ---
        cl_raw = float(self.model_cl.predict(x)[0])
        cd_raw = float(self.model_cd.predict(x)[0])

        # --- Output sanity checks (flag, don't silently fix — physics module handles clamping) ---
        if not (_CL_RAW_MIN <= cl_raw <= _CL_RAW_MAX):
            warnings.append(f"Raw Cl={cl_raw:.4f} outside expected range [{_CL_RAW_MIN}, {_CL_RAW_MAX}]")
            cl_raw = np.clip(cl_raw, _CL_RAW_MIN, _CL_RAW_MAX)

        if cd_raw < _CD_RAW_MIN:
            warnings.append(f"Raw Cd={cd_raw:.6f} is negative — clamping to 0")
            cd_raw = 0.0  # Physics module will enforce minimum drag

        if cd_raw > _CD_RAW_MAX:
            warnings.append(f"Raw Cd={cd_raw:.4f} unusually high — possible extrapolation")
            cd_raw = np.clip(cd_raw, 0.0, _CD_RAW_MAX)

        return AirfoilPrediction(
            cl=cl_raw,
            cd=cd_raw,
            is_extrapolated=is_extrapolated,
            warnings=warnings,
        )

    def predict_batch(
        self,
        re_array: np.ndarray,
        aoa_array: np.ndarray,
        thickness: float,
        thickness_loc: float,
        camber: float,
        camber_loc: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for arrays of Re and AoA (same geometry).

        Used for spanwise station sweeps where airfoil shape is constant
        but Re and effective AoA vary along the span.

        Args:
            re_array: 1D array of Reynolds numbers.
            aoa_array: 1D array of angles of attack (degrees).
            thickness, thickness_loc, camber, camber_loc: Airfoil geometry (scalar).

        Returns:
            Tuple of (cl_array, cd_array).
        """
        n = len(re_array)
        if len(aoa_array) != n:
            raise ValueError(
                f"re_array length ({n}) != aoa_array length ({len(aoa_array)})"
            )

        # Construct feature matrix [n_samples, 6]
        X = np.column_stack([
            re_array,
            aoa_array,
            np.full(n, thickness),
            np.full(n, thickness_loc),
            np.full(n, camber),
            np.full(n, camber_loc),
        ])

        cl_batch = self.model_cl.predict(X).astype(float)
        cd_batch = self.model_cd.predict(X).astype(float)

        # Sanity clamp
        cl_batch = np.clip(cl_batch, _CL_RAW_MIN, _CL_RAW_MAX)
        cd_batch = np.clip(cd_batch, 0.0, _CD_RAW_MAX)

        return cl_batch, cd_batch
