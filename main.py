import numpy as np
import os
from core.airfoil_ml import AirfoilML
from core.physics import enforce_physics
from core.llt import solve_llt
from core.validation import validate_results
from core.config import PRESETS
from utils.helpers import get_atmosphere, plot_aerodynamics

def run_analysis(wing_type='general_aviation', N=40):
    print(f"\n--- Wing Analyzer: {wing_type.upper()} ---")
    
    # 1. Load configuration and atmosphere
    config = PRESETS[wing_type]
    atm = get_atmosphere(config['altitude'])
    rho, mu, V = atm['rho'], atm['mu'], config['cruise_speed']
    
    # 2. Instantiate ML model
    models_dir = os.path.join(os.getcwd(), 'models')
    ml_model = AirfoilML(models_dir)
    
    # 3. Discretize wing spanwise
    theta = np.linspace(np.pi/(2*N), np.pi/2, N)
    y_stations = (config['span']/2) * np.cos(theta)
    eta_stations = y_stations / (config['span']/2)
    
    chord_stations = config['root_chord'] + (config['tip_chord'] - config['root_chord']) * (1 - eta_stations)
    re_stations = rho * V * chord_stations / mu
    
    # 4. Extract section properties distribution (a0 and alpha_0)
    print("Calculating section properties distribution using ML model...")
    a0_dist = []
    al0_dist = []
    
    aoa_test_range = np.arange(-5, 6, 1) 
    
    for i in range(N):
        cl_samples = []
        for aoa in aoa_test_range:
            cl, _ = ml_model.predict(
                re_stations[i], aoa, 
                config['thickness'], config['thickness_loc'], 
                config['camber'], config['camber_loc']
                # Removed te_thickness and le_thickness as per repo model requirement
            )
            cl_samples.append(cl)
        
        # Fit slope (a0) in radians
        slope, _ = np.polyfit(np.radians(aoa_test_range), cl_samples, 1)
        a0_dist.append(slope)
        
        # Interpolate to find zero-lift angle (al0)
        al0 = np.interp(0.0, cl_samples, np.radians(aoa_test_range))
        al0_dist.append(al0)
    
    a0_dist = np.array(a0_dist)
    al0_dist = np.array(al0_dist)
    
    # 5. Global AoA Sweep
    aoa_sweep = np.arange(-4, 15, 1)
    cl_results = []
    cdi_results = []
    sample_cl_dist = None
    y_plot_dist = None
    
    print("Performing global wing sweep...")
    for aoa in aoa_sweep:
        res = solve_llt(
            N, aoa, config['span'], config['root_chord'], config['tip_chord'], 
            config['twist_deg'], a0_dist, al0_dist
        )
        
        if res:
            cl_results.append(res['CL'])
            cdi_results.append(res['CDi'])
            
            if aoa == 5:
                sample_cl_dist = res['cl_dist']
                y_plot_dist = res['y_dist']
    
    # 6. Basic Parasite Drag (CD0) estimate
    ff = 1.0 + (0.6 / config['thickness_loc']) * config['thickness'] + 100 * config['thickness']**4
    c_mac = (config['root_chord'] + config['tip_chord']) / 2
    re_mac = rho * V * c_mac / mu
    cf = 0.455 / (np.log10(max(re_mac, 1e5)))**2.58
    cd0 = cf * ff * 1.2
    
    cd_total = np.array(cdi_results) + cd0
    
    # 7. Package results
    results = {
        'wing_type': wing_type,
        'aoa_sweep': aoa_sweep,
        'cl_sweep': np.array(cl_results),
        'cd_sweep': cd_total,
        'y_dist': y_plot_dist,
        'cl_dist': sample_cl_dist
    }
    
    # 8. Final Validation
    max_cl = np.max(results['cl_sweep'])
    max_cd = np.max(results['cd_sweep'])
    validate_results(max_cl, max_cd)
    
    print(f"Analysis Complete. Max CL: {max_cl:.3f}, CD0: {cd0:.5f}")
    
    # 9. Plotting
    plot_aerodynamics(results)

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
        
    run_analysis('general_aviation')
