import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Attentional Superposition Simulator",
    layout="wide",
    page_icon="⚡",
)

st.title("⚡ Dynamic Real-Time Superposition Mechanics")
st.caption(
    "Live Simulation of Query-Key Interference & Phase Transition | IRIS Fair 2026-2027"
)

# Sidebar Setup
st.sidebar.header("⚙️ Simulation Controls")
d_head = st.sidebar.slider("Head Dimension (d_head)", 8, 64, 16, step=8)
num_steps = st.sidebar.slider("Animation Resolution Steps", 10, 50, 20)
run_sim = st.sidebar.button("▶️ Run Real-Time Phase Transition Simulation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Real-Time Attention Interference Matrix")
    matrix_placeholder = st.empty()

with col2:
    st.subheader("2. Live Singular Value Decay & Rank Truncation")
    spectrum_placeholder = st.empty()

st.subheader("3. Dynamic Softmax Entropy & Cross-Talk Explosion")
softmax_placeholder = st.empty()

if run_sim or st.sidebar.checkbox("Auto-Run Live Loop", value=True):
    # Dynamic Simulation Loop
    n_ratios = np.linspace(0.2, 2.5, num_steps)

    for ratio in n_ratios:
        N = int(d_head * ratio)

        # 1. Synthesize random feature projections
        Q = np.random.randn(N, d_head) / np.sqrt(d_head)
        K = np.random.randn(N, d_head) / np.sqrt(d_head)

        # Raw Score Matrix
        S = np.dot(Q, K.T)

        # 2. Compute SVD for Singular Value Spectrum
        U, s, Vh = np.linalg.svd(S)

        # 3. Compute Softmax Attention Distribution
        S_softmax = np.exp(S - np.max(S, axis=-1, keepdims=True))
        S_softmax /= np.sum(S_softmax, axis=-1, keepdims=True)

        # --- Plot 1: Dynamic Heatmap of Score Matrix ---
        fig_mat = px.imshow(
            S,
            color_continuous_scale="Viridis",
            title=f"Score Matrix Score (N={N}, d_head={d_head}) | Load N/d = {ratio:.2f}",
            labels=dict(x="Key Index", y="Query Index", color="Attention Score"),
        )
        fig_mat.update_layout(
            template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10)
        )
        matrix_placeholder.plotly_chart(fig_mat, use_container_width=True)

        # --- Plot 2: Live Singular Value Decay ---
        fig_svd = go.Figure()
        fig_svd.add_trace(
            go.Scatter(
                y=s,
                mode="lines+markers",
                marker=dict(size=6, color="#00FFA3"),
                line=dict(width=3),
                name="Singular Values",
            )
        )
        fig_svd.add_vline(
            x=min(d_head, N) - 1,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Effective Rank Boundary ({min(d_head, N)})",
        )
        fig_svd.update_layout(
            template="plotly_dark",
            height=380,
            xaxis_title="Singular Value Index",
            yaxis_title="Magnitude",
            title=f"Singular Spectrum Spectrum (Rank Max = {min(N, d_head)})",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        spectrum_placeholder.plotly_chart(
            fig_svd, use_container_width=True
        )

        # --- Plot 3: Softmax Distribution Interference ---
        fig_soft = px.imshow(
            S_softmax,
            color_continuous_scale="magma",
            title=f"Softmax Cross-Talk Noise Spread (N/d = {ratio:.2f})",
        )
        fig_soft.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        softmax_placeholder.plotly_chart(fig_soft, use_container_width=True)

        time.sleep(0.15)  # Smooth animation feel
