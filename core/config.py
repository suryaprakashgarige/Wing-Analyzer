# Configuration for Wing Analyzer
# Ported from Wing_Analysis_FIXED (2).ipynb

PRESETS = {
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

# Physical Constants
GRAVITY = 9.80665
R_AIR = 287.05
GAMMA = 1.4
T_SL = 288.15
P_SL = 101325.0
L_LAPSE = 0.0065
