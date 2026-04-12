import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from core.airfoil_ml import AirfoilML
from core.llt import solve_llt
from core.config import PRESETS
from utils.helpers import get_atmosphere

# --- Page Config ---
st.set_page_config(page_title="Wing Analyzer Pro", layout="wide", page_icon="✈️")

# --- Premium Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 { color: #1e293b; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# --- App State ---
if 'ml_model' not in st.session_state:
    models_dir = os.path.join(os.getcwd(), 'models')
    st.session_state.ml_model = AirfoilML(models_dir)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/aircraft-wing.png", width=80)
    st.title("Wing Config")
    
    selected_preset = st.selectbox("Load Preset", list(PRESETS.keys()), index=1)
    p = PRESETS[selected_preset]
    
    st.divider()
    
    span = st.slider("Wingspan (m)", 1.0, 80.0, float(p['span']), 0.5)
    root_c = st.slider("Root Chord (m)", 0.1, 10.0, float(p['root_chord']), 0.1)
    tip_c = st.slider("Tip Chord (m)", 0.1, 10.0, float(p['tip_chord']), 0.1)
    sweep = st.slider("Sweep (deg)", 0.0, 45.0, float(p.get('sweep_deg', 0.0)), 0.5)
    twist = st.slider("Twist/Washout (deg)", -10.0, 5.0, float(p['twist_deg']), 0.5)
    
    st.divider()
    
    alt = st.number_input("Altitude (m)", 0, 15000, int(p['altitude']), 100)
    speed = st.number_input("Cruise Speed (m/s)", 10, 500, int(p['cruise_speed']), 5)

# --- Logic Bridge ---
def compute_aerodynamics():
    # 1. Atmosphere
    atm = get_atmosphere(alt)
    rho, mu = atm['rho'], atm['mu']
    
    # 2. Section Props Distribution
    N = 40
    theta = np.linspace(np.pi/(2*N), np.pi/2, N)
    y_stations = (span/2) * np.cos(theta)
    eta = y_stations / (span/2)
    chords = root_c + (tip_c - root_c) * (1 - eta)
    res = rho * speed * chords / mu
    
    a0_dist = []
    al0_dist = []
    
    # Fast ML predict loop
    aoa_test = np.array([-4, 0, 4])
    for i in range(N):
        cl_samples = []
        for a in aoa_test:
            cl, _ = st.session_state.ml_model.predict(
                res[i], a, p['thickness'], p['thickness_loc'], p['camber'], p['camber_loc']
            )
            cl_samples.append(cl)
        
        slope, _ = np.polyfit(np.radians(aoa_test), cl_samples, 1)
        al0 = np.interp(0.0, cl_samples, np.radians(aoa_test))
        a0_dist.append(slope)
        al0_dist.append(al0)

    # 3. Global Sweep
    aoa_sweep = np.arange(-4, 15, 1)
    results_list = []
    sample_cl_dist = []
    
    for aoa in aoa_sweep:
        run = solve_llt(N, aoa, span, root_c, tip_c, twist, np.array(a0_dist), np.array(al0_dist))
        if run:
            results_list.append({'aoa': aoa, 'CL': run['CL'], 'CDi': run['CDi']})
            if aoa == 5:
                sample_cl_dist = run['cl_dist']

    # 4. Drag & Metrics
    df = pd.DataFrame(results_list)
    # CD0 estimate
    ff = 1.0 + (0.6 / p['thickness_loc']) * p['thickness'] + 100 * p['thickness']**4
    cf = 0.455 / (np.log10(max(rho * speed * ((root_c+tip_c)/2) / mu, 1e5)))**2.58
    cd0 = cf * ff * 1.2
    df['CD'] = df['CDi'] + cd0
    df['LD'] = df['CL'] / df['CD']
    
    return df, y_stations, sample_cl_dist

# --- Main UI ---
st.title("Wing Analyzer Pro")
st.markdown(f"**Analysis Mode:** Physics-Informed ML + Lifting Line Theory | **Configuration:** {selected_preset.replace('_',' ').title()}")

if st.button("RUN ANALYSIS", type="primary"):
    with st.spinner("Calculating aerodynamic matrices..."):
        df, y_s, cl_dist = compute_aerodynamics()
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        best_ld = df['LD'].max()
        best_aoa = df.loc[df['LD'].idxmax(), 'aoa']
        m1.metric("Max Efficiency (L/D)", f"{best_ld:.1f}")
        m2.metric("Optimal AoA", f"{best_aoa:.1f}°")
        m3.metric("Max Wing CL", f"{df['CL'].max():.3f}")
        m4.metric("Parasite Drag (CD0)", f"{df['CD'].min() - df['CDi'].min():.5f}")
        
        st.divider()
        
        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            fig_lift = px.line(df, x="aoa", y="CL", title="Wing Lift Curve (CL vs α)", template="plotly_white")
            fig_lift.update_traces(line_color='#1d4ed8', line_width=3)
            st.plotly_chart(fig_lift, use_container_width=True)
            
            fig_polar = px.line(df, x="CD", y="CL", title="Drag Polar (CL vs CD)", template="plotly_white")
            fig_polar.update_traces(line_color='#059669', line_width=3)
            st.plotly_chart(fig_polar, use_container_width=True)
            
        with c2:
            fig_ld = px.line(df, x="aoa", y="LD", title="Lift-to-Drag Ratio (Efficiency)", template="plotly_white")
            fig_ld.update_traces(line_color='#7c3aed', line_width=3)
            st.plotly_chart(fig_ld, use_container_width=True)
            
            fig_dist = px.area(x=y_s, y=cl_dist, title="Spanwise Lift Distribution (@ 5° AoA)", 
                             labels={'x': 'Span Position (m)', 'y': 'Local Cl'}, template="plotly_white")
            fig_dist.update_traces(fillcolor='rgba(29, 78, 216, 0.2)', line_color='#1d4ed8')
            st.plotly_chart(fig_dist, use_container_width=True)
            
else:
    st.info("Adjust the parameters in the sidebar and click **RUN ANALYSIS** to visualize the aerodynamic performance.")
