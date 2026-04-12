import numpy as np
import matplotlib.pyplot as plt
import os
from core.config import GRAVITY, R_AIR, GAMMA, T_SL, P_SL, L_LAPSE

def get_atmosphere(altitude_m):
    """
    ISA Atmosphere Model.
    """
    # Clip altitude to troposphere for basic ISA
    alt = min(altitude_m, 11000)
    
    t = T_SL - L_LAPSE * alt
    p = P_SL * (t / T_SL) ** (GRAVITY / (L_LAPSE * R_AIR))
    rho = p / (R_AIR * t)
    
    # Sutherland's law for viscosity
    mu0 = 1.716e-5
    s_sl = 110.4
    mu = mu0 * (t / 273.15)**1.5 * (273.15 + s_sl) / (t + s_sl)
    
    a = np.sqrt(GAMMA * R_AIR * t)
    
    return {
        'rho': rho,
        't': t,
        'p': p,
        'mu': mu,
        'a': a
    }

def plot_aerodynamics(results, output_dir="results"):
    """
    Generates premium aerodynamic plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.style.use('bmh') # Clean readable style
    
    # 1. CL vs Alpha
    plt.figure(figsize=(10, 6))
    plt.plot(results['aoa_sweep'], results['cl_sweep'], 'o-', linewidth=2, color='#1D4ED8', label='Wing CL')
    plt.xlabel('Angle of Attack (deg)', fontsize=12)
    plt.ylabel('Lift Coefficient (CL)', fontsize=12)
    plt.title('Wing Lift Curve (CL vs Alpha)', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cl_vs_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spanwise Lift Distribution (using the test result or a specific AoA point)
    if 'y_dist' in results and 'cl_dist' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['y_dist'], results['cl_dist'], linewidth=2, color='#DC2626', label='Local Cl')
        plt.xlabel('Spanwise Position (y)', fontsize=12)
        plt.ylabel('Local Lift Coefficient (Cl)', fontsize=12)
        plt.title('Spanwise Lift Distribution', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'spanwise_lift.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 3. Drag Polar
    if 'cd_sweep' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['cd_sweep'], results['cl_sweep'], 'o-', linewidth=2, color='#059669', label='Wing Polar')
        plt.xlabel('Drag Coefficient (CD)', fontsize=12)
        plt.ylabel('Lift Coefficient (CL)', fontsize=12)
        plt.title('Wing Drag Polar (CL vs CD)', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'drag_polar.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}/")
