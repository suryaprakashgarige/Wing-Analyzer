# app.py
"""
Wing Analyzer Pro -- Streamlit UI.

Pipeline (matches main.py exactly):
  ML -> OOD check -> physics clamp -> lift slope extraction -> LLT -> drag model -> validation

All computation delegated to core modules. This is presentation only.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from core.airfoil_ml import AirfoilML, compute_lift_slope, compute_zero_lift_angle
from core.physics import enforce_physics
from core.llt import solve_llt, LLTError
from core.drag import compute_wing_drag
from core.ood import check_prediction_ood
from core.validation import (
    validate_wing_coefficients,
    validate_spanwise_distribution,
    Severity,
)
from core.config import PRESETS
from utils.helpers import get_atmosphere


# --- Page Config ---
st.set_page_config(page_title="Wing Analyzer Pro", layout="wide", page_icon="?")

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


def plot_wing_3d(span, chord_root, chord_tip, N=30):
    y = np.linspace(-span/2, span/2, N)
    chord = chord_root + (chord_tip - chord_root) * (np.abs(y)/(span/2))

    x_le = np.zeros(N)
    x_te = chord

    fig = go.Figure()

    # Leading edge
    fig.add_trace(go.Scatter3d(x=x_le, y=y, z=np.zeros(N),
                              mode='lines', name='Leading Edge',
                              line=dict(color='#1d4ed8', width=4)))

    # Trailing edge
    fig.add_trace(go.Scatter3d(x=x_te, y=y, z=np.zeros(N),
                              mode='lines', name='Trailing Edge',
                              line=dict(color='#059669', width=4)))

    # Surface (simple mesh)
    X = np.array([x_le, x_te])
    Y = np.array([y, y])
    Z = np.zeros_like(X)

    fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.4, colorscale='Blues'))

    fig.update_layout(
        title="3D Wing Geometry",
        scene=dict(
            xaxis_title="Chord (m)",
            yaxis_title="Span (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=450
    )

    return fig


# --- Model Loading (cached) ---
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


# --- Computation (same pipeline as main.py) ---
def compute_aerodynamics():
    """
    Pipeline: ML -> OOD check -> lift slope extraction -> LLT -> drag model -> validation
    """
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

    # 3. OOD check + section properties via central-difference a0
    a0_dist = np.empty(N)
    al0_dist = np.empty(N)
    ood_count = 0

    for i in range(N):
        # OOD check
        ood = check_prediction_ood(
            re_stations[i], 0.0,
            p['thickness'], p['thickness_loc'],
            p['camber'], p['camber_loc'],
        )

        if not ood.is_in_distribution:
            a0_dist[i] = 2.0 * np.pi
            al0_dist[i] = -2.0 * p['camber']
            ood_count += 1
            continue

        # Central-difference lift slope (clamped [4.5, 7.0])
        a0_dist[i] = compute_lift_slope(
            ml_model, 0.0, re_stations[i],
            p['thickness'], p['thickness_loc'],
            p['camber'], p['camber_loc'],
        )

        # Zero-lift angle via interpolation
        al0_dist[i] = compute_zero_lift_angle(
            ml_model, re_stations[i],
            p['thickness'], p['thickness_loc'],
            p['camber'], p['camber_loc'],
        )

    # 4. AoA sweep with LLT
    aoa_sweep = np.arange(-4.0, 16.0, 1.0)
    results_list = []
    sample_cl_dist = None
    sample_y_dist = None

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

    if not results_list:
        return None, None, None, None, 0

    df = pd.DataFrame(results_list)

    # 5. Drag model: CD = CD0 + CDi (NOT ML Cd)
    AR = span ** 2 / (0.5 * (root_c + tip_c) * span)
    drag_ref = compute_wing_drag(
        CL=0.5, AR=AR, e=df['e'].max(),
        rho=rho, V=speed, mu=mu,
        root_chord=root_c, tip_chord=tip_c, span=span,
        thickness=p['thickness'], thickness_loc=p['thickness_loc'],
    )
    cd0 = drag_ref.CD0

    df['CD'] = df['CDi'] + cd0
    df['LD'] = df['CL'] / df['CD']
    df['cd0'] = cd0

    # 6. Validation
    report = validate_wing_coefficients(
        float(df['CL'].max()), float(df['CD'].max()),
        CDi=float(df['CDi'].max()),
        e=float(df['e'].min()),
        AR=AR,
    )

    return df, sample_y_dist, sample_cl_dist, report, ood_count


# --- Main UI ---
st.title("Wing Analyzer Pro")
st.markdown(
    f"**Pipeline:** ML -> OOD Check -> Physics Clamp -> LLT -> Drag Model (CD0+CDi) -> Validation | "
    f"**Config:** {selected_preset.replace('_', ' ').title()}"
)

# 3D Wing Geometry (Always visible)
st.plotly_chart(plot_wing_3d(span, root_c, tip_c), use_container_width=True)

if st.button("RUN ANALYSIS", type="primary"):
    with st.spinner("Running physics-constrained analysis pipeline..."):
        try:
            df, y_s, cl_dist, report, ood_count = compute_aerodynamics()
        except Exception as e:
            st.error(f"Analysis failed with solver error: {str(e)}")
            st.stop()

        if df is None:
            st.error("Analysis failed -- LLT solver could not converge at any AoA.")
        else:
            # --- OOD Status ---
            if ood_count > 0:
                st.warning(
                    f"{ood_count}/{N_stations} spanwise stations were outside ML training "
                    f"envelope and used thin-airfoil fallback."
                )

            # --- Validation Status ---
            if report and not report.is_valid:
                for err in report.errors:
                    st.error(f"{err.parameter}: {err.message}")
            elif report and report.warnings:
                for w in report.warnings:
                    st.warning(f"{w.parameter}: {w.message}")

            # --- Top Metrics ---
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            best_ld = df['LD'].max()
            best_aoa = df.loc[df['LD'].idxmax(), 'aoa']
            cd0 = df['cd0'].iloc[0]
            cdi_at_best_ld = df.loc[df['LD'].idxmax(), 'CDi']
            
            MAC = (2/3) * root_c * (1 + (tip_c/root_c) + (tip_c/root_c)**2) / (1 + (tip_c/root_c))
            atm = get_atmosphere(alt)
            Re_mac = atm['rho'] * speed * MAC / atm['mu']

            m1.metric("Max CL", f"{df['CL'].max():.3f}")
            m2.metric("Min CD", f"{df['CD'].min():.4f}")
            m3.metric("Max L/D", f"{best_ld:.1f}")
            m4.metric("CD0 (Parasite)", f"{cd0:.4f}")
            m5.metric("CDi (@ Max L/D)", f"{cdi_at_best_ld:.4f}")
            m6.metric("Re (MAC)", f"{Re_mac/1e6:.2f}M")

            st.divider()

            # --- Charts ---
            c1, c2 = st.columns(2)

            with c1:
                fig_lift = px.line(
                    df, x="aoa", y="CL",
                    title="Wing Lift Curve (CL vs AoA)",
                    template="plotly_white",
                )
                fig_lift.update_traces(line_color='#1d4ed8', line_width=3)
                fig_lift.update_layout(
                    xaxis_title="Angle of Attack (deg)",
                    yaxis_title="CL",
                )
                st.plotly_chart(fig_lift, use_container_width=True)

                fig_polar = px.line(
                    df, x="CD", y="CL",
                    title="Drag Polar (CL vs CD = CD0 + CDi)",
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
                    xaxis_title="Angle of Attack (deg)",
                    yaxis_title="L/D",
                )
                st.plotly_chart(fig_ld, use_container_width=True)

                if y_s is not None and cl_dist is not None:
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
                        title="Spanwise Lift Distribution (@ 5 deg AoA)",
                        xaxis_title="Span Position (m)",
                        yaxis_title="Local Cl",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            # --- Detailed Data ---
            with st.expander("Detailed Results Table"):
                st.dataframe(
                    df[['aoa', 'CL', 'CDi', 'CD', 'LD', 'e', 'cond']].round(5),
                    use_container_width=True,
                )

            # --- Validation Report ---
            if report and report.issues:
                with st.expander("Validation Report"):
                    for issue in report.issues:
                        icon = {"INFO": "i", "WARNING": "!", "ERROR": "X"}
                        sev = issue.severity.value
                        val = f" = {issue.value:.4f}" if issue.value else ""
                        st.markdown(
                            f"**[{sev}]** {issue.parameter}{val}: {issue.message}"
                        )

else:
    st.info(
        "Adjust the parameters in the sidebar and click **RUN ANALYSIS** "
        "to run the physics-constrained analysis pipeline."
    )
