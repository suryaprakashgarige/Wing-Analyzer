# app.py
"""
Wing Analyzer Pro — Streamlit UI.

This is the presentation layer only. All computation is delegated to:
  - core/airfoil_ml.py (ML predictions)
  - core/physics.py (constraint enforcement)
  - core/llt.py (lifting line theory)
  - core/validation.py (result validation)
  - utils/helpers.py (atmosphere, drag estimation)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from core.airfoil_ml import AirfoilML
from core.physics import enforce_physics
from core.llt import solve_llt, LLTError
from core.validation import (
    validate_wing_coefficients,
    validate_spanwise_distribution,
    Severity,
)
from core.config import PRESETS
from utils.helpers import get_atmosphere, estimate_parasite_drag


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


# --- Model Loading (cached in session) ---
@st.cache_resource
def load_ml_model():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    return AirfoilML(models_dir)


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

    st.divider()
    N_stations = st.slider("LLT Stations", 10, 100, 40, 5)


# --- Computation ---
def compute_aerodynamics():
    """Run the full analysis pipeline."""
    ml_model = load_ml_model()

    # 1. Atmosphere
    atm = get_atmosphere(alt)
    rho, mu = atm['rho'], atm['mu']

    # 2. Spanwise discretization
    N = N_stations
    theta = np.linspace(np.pi / (2 * N), np.pi / 2, N)
    y_stations = (span / 2) * np.cos(theta)
    eta = y_stations / (span / 2)
    chords = root_c + (tip_c - root_c) * (1 - eta)
    re_stations = rho * speed * chords / mu

    # 3. Section properties via ML + physics
    a0_dist = np.empty(N)
    al0_dist = np.empty(N)

    aoa_test = np.arange(-5.0, 6.0, 1.0)
    aoa_test_rad = np.radians(aoa_test)

    for i in range(N):
        cl_samples = np.empty(len(aoa_test))
        for k, aoa_val in enumerate(aoa_test):
            cl, _ = ml_model.predict(
                re_stations[i], aoa_val,
                p['thickness'], p['thickness_loc'],
                p['camber'], p['camber_loc'],
            )
            cl, _ = enforce_physics(
                aoa_val, cl, 0.01,
                p['thickness'], p['camber'],
            )
            cl_samples[k] = cl

        coeffs = np.polyfit(aoa_test_rad, cl_samples, 1)
        a0 = coeffs[0]
        b = coeffs[1]

        if abs(a0) > 0.1:
            al0 = -b / a0
        else:
            al0 = -2.0 * p['camber']
            a0 = 2.0 * np.pi

        a0_dist[i] = a0
        al0_dist[i] = al0

    # 4. AoA sweep with LLT
    aoa_sweep = np.arange(-4.0, 16.0, 1.0)
    results_list = []
    sample_cl_dist = None
    sample_y_dist = None
    sample_alpha_i = None

    for aoa in aoa_sweep:
        try:
            res = solve_llt(
                N, aoa, span, root_c, tip_c, twist,
                a0_dist, al0_dist,
            )
        except LLTError:
            continue

        if res is not None:
            results_list.append({
                'aoa': aoa,
                'CL': res.CL,
                'CDi': res.CDi,
                'e': res.e,
                'cond': res.cond_number,
            })

            if abs(aoa - 5.0) < 0.5:
                sample_cl_dist = res.cl_dist
                sample_y_dist = res.y_dist
                sample_alpha_i = res.alpha_i_dist

    if not results_list:
        return None, None, None, None

    df = pd.DataFrame(results_list)

    # 5. Parasite drag
    cd0 = estimate_parasite_drag(
        rho, speed, root_c, tip_c, span, mu,
        p['thickness'], p['thickness_loc'],
    )

    df['CD'] = df['CDi'] + cd0
    df['LD'] = df['CL'] / df['CD']
    df['cd0'] = cd0

    # 6. Validation
    max_cl = df['CL'].max()
    max_cd = df['CD'].max()
    report = validate_wing_coefficients(
        max_cl, max_cd,
        CDi=df['CDi'].max(),
        e=df['e'].min(),
    )

    return df, sample_y_dist, sample_cl_dist, report


# --- Main UI ---
st.title("Wing Analyzer Pro")
st.markdown(
    f"**Analysis Mode:** Physics-Informed ML + Lifting Line Theory | "
    f"**Configuration:** {selected_preset.replace('_', ' ').title()}"
)

if st.button("RUN ANALYSIS", type="primary"):
    with st.spinner("Calculating aerodynamic matrices..."):
        df, y_s, cl_dist, report = compute_aerodynamics()

        if df is None:
            st.error("Analysis failed — LLT solver could not converge at any AoA.")
        else:
            # --- Validation Status ---
            if report and not report.is_valid:
                for err in report.errors:
                    st.error(f"⚠ {err.parameter}: {err.message}")
            elif report and report.warnings:
                for w in report.warnings:
                    st.warning(f"⚠ {w.parameter}: {w.message}")

            # --- Top Metrics ---
            m1, m2, m3, m4 = st.columns(4)
            best_ld = df['LD'].max()
            best_aoa = df.loc[df['LD'].idxmax(), 'aoa']
            cd0 = df['cd0'].iloc[0]

            m1.metric("Max Efficiency (L/D)", f"{best_ld:.1f}")
            m2.metric("Optimal AoA", f"{best_aoa:.1f}°")
            m3.metric("Max Wing CL", f"{df['CL'].max():.3f}")
            m4.metric("Parasite Drag (CD0)", f"{cd0:.5f}")

            st.divider()

            # --- Charts ---
            c1, c2 = st.columns(2)

            with c1:
                fig_lift = px.line(
                    df, x="aoa", y="CL",
                    title="Wing Lift Curve (CL vs α)",
                    template="plotly_white",
                )
                fig_lift.update_traces(line_color='#1d4ed8', line_width=3)
                fig_lift.update_layout(
                    xaxis_title="Angle of Attack (°)",
                    yaxis_title="CL",
                )
                st.plotly_chart(fig_lift, use_container_width=True)

                fig_polar = px.line(
                    df, x="CD", y="CL",
                    title="Drag Polar (CL vs CD)",
                    template="plotly_white",
                )
                fig_polar.update_traces(line_color='#059669', line_width=3)
                st.plotly_chart(fig_polar, use_container_width=True)

            with c2:
                fig_ld = px.line(
                    df, x="aoa", y="LD",
                    title="Lift-to-Drag Ratio (Efficiency)",
                    template="plotly_white",
                )
                fig_ld.update_traces(line_color='#7c3aed', line_width=3)
                fig_ld.update_layout(
                    xaxis_title="Angle of Attack (°)",
                    yaxis_title="L/D",
                )
                st.plotly_chart(fig_ld, use_container_width=True)

                if y_s is not None and cl_dist is not None:
                    # Mirror for full span
                    y_full = np.concatenate([-y_s[::-1], y_s])
                    cl_full = np.concatenate([cl_dist[::-1], cl_dist])

                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Scatter(
                        x=y_full, y=cl_full,
                        mode='lines',
                        fill='tozeroy',
                        fillcolor='rgba(29, 78, 216, 0.15)',
                        line=dict(color='#1d4ed8', width=2.5),
                        name='Local Cl',
                    ))
                    fig_dist.update_layout(
                        title="Spanwise Lift Distribution (@ 5° AoA)",
                        xaxis_title="Span Position (m)",
                        yaxis_title="Local Cl",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            # --- Detailed Data ---
            with st.expander("📊 Detailed Results Table"):
                st.dataframe(
                    df[['aoa', 'CL', 'CDi', 'CD', 'LD', 'e', 'cond']].round(5),
                    use_container_width=True,
                )

            # --- Validation Report ---
            if report and report.issues:
                with st.expander("🔍 Validation Report"):
                    for issue in report.issues:
                        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}
                        sev = issue.severity.value
                        val = f" = {issue.value:.4f}" if issue.value else ""
                        st.markdown(
                            f"{icon.get(sev, '•')} **[{sev}]** {issue.parameter}{val}: "
                            f"{issue.message}"
                        )

else:
    st.info(
        "Adjust the parameters in the sidebar and click **RUN ANALYSIS** "
        "to visualize the aerodynamic performance."
    )
