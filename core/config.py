# core/config.py
"""
Wing Analyzer Configuration.

Centralized aircraft presets and physical constants.
No computed values here — only raw configuration data.
"""

from typing import Dict, Any

# --- Physical Constants (ISA Standard Atmosphere) ---
GRAVITY = 9.80665       # m/s² — standard gravitational acceleration
R_AIR = 287.05          # J/(kg·K) — specific gas constant for dry air
GAMMA = 1.4             # — ratio of specific heats for air
T_SL = 288.15           # K — sea-level standard temperature
P_SL = 101325.0         # Pa — sea-level standard pressure
L_LAPSE = 0.0065        # K/m — tropospheric temperature lapse rate
MU_REF = 1.716e-5       # Pa·s — reference dynamic viscosity (Sutherland)
T_SUTH = 110.4          # K — Sutherland's temperature constant


# --- Aircraft Presets ---
# Each preset defines a complete wing + flight condition.
#
# Airfoil parameters (thickness, camber, etc.) correspond to the
# training data of the ML models. Changing these beyond the model's
# training envelope will trigger extrapolation warnings in airfoil_ml.py.

PRESETS: Dict[str, Dict[str, Any]] = {
    'UAV': {
        'label': 'UAV / Drone',
        'span': 2.0,
        'root_chord': 0.25,
        'tip_chord': 0.15,
        'sweep_deg': 0.0,
        'twist_deg': -2.0,
        'dihedral_deg': 3.0,
        'cruise_speed': 20.0,
        'altitude': 100.0,
        'MTOW_kg': 2.5,
        # Airfoil geometry (NACA-like parameters)
        'thickness': 0.12,
        'thickness_loc': 0.30,
        'camber': 0.02,
        'camber_loc': 0.40,
    },
    'general_aviation': {
        'label': 'General Aviation',
        'span': 11.0,
        'root_chord': 1.6,
        'tip_chord': 0.9,
        'sweep_deg': 3.0,
        'twist_deg': -2.0,
        'dihedral_deg': 5.0,
        'cruise_speed': 55.0,
        'altitude': 3000.0,
        'MTOW_kg': 1100.0,
        'thickness': 0.12,
        'thickness_loc': 0.30,
        'camber': 0.02,
        'camber_loc': 0.40,
    },
    'glider': {
        'label': 'Glider / Sailplane',
        'span': 18.0,
        'root_chord': 0.90,
        'tip_chord': 0.35,
        'sweep_deg': 1.0,
        'twist_deg': -3.0,
        'dihedral_deg': 4.0,
        'cruise_speed': 28.0,
        'altitude': 1500.0,
        'MTOW_kg': 320.0,
        'thickness': 0.14,
        'thickness_loc': 0.35,
        'camber': 0.035,
        'camber_loc': 0.45,
    },
    'fighter': {
        'label': 'Fighter / High-Speed',
        'span': 9.0,
        'root_chord': 4.5,
        'tip_chord': 1.0,
        'sweep_deg': 35.0,
        'twist_deg': 0.0,
        'dihedral_deg': 0.0,
        'cruise_speed': 250.0,
        'altitude': 10000.0,
        'MTOW_kg': 12000.0,
        'thickness': 0.06,
        'thickness_loc': 0.40,
        'camber': 0.005,
        'camber_loc': 0.50,
    },
    'transport': {
        'label': 'Transport / Commercial',
        'span': 35.0,
        'root_chord': 6.0,
        'tip_chord': 2.0,
        'sweep_deg': 25.0,
        'twist_deg': -3.0,
        'dihedral_deg': 6.0,
        'cruise_speed': 240.0,
        'altitude': 11000.0,
        'MTOW_kg': 75000.0,
        'thickness': 0.13,
        'thickness_loc': 0.35,
        'camber': 0.03,
        'camber_loc': 0.45,
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Retrieve a preset by name with validation.

    Args:
        name: Preset key (e.g. 'general_aviation').

    Returns:
        Configuration dictionary.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
