import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Neural Attention Pipeline Flow", layout="wide"
)

st.title("⚡ Dynamic Neural Network Pipeline: Attention Superposition Collapse")
st.caption("IRIS National Fair | Automated End-to-End Method & Result Simulation")

run_btn = st.button("▶️ Start Auto Pipeline (1-Click Full Demo)")

status_box = st.empty()
plot_placeholder = st.empty()

if run_btn:
    # Define Network Pipeline Nodes (X, Y coordinates for neural flow)
    nodes_x = [1, 2, 3, 4, 5]
    nodes_y = [2, 2, 2, 2, 2]
    node_labels = [
        "1. Token Inputs",
        "2. QK Dot-Product",
        "3. Softmax Amplification",
        "4. Pythia-70M SVD",
        "5. ETF Geometry Fix",
    ]

    # Automated Simulation Timeline
    stages = [
        {
            "load": 0.5,
            "status": "🟢 STAGE 1: Normal Token Embedding (N/d = 0.5)",
            "color": "#00FFA3",
            "active_node": 0,
        },
        {
            "load": 1.0,
            "status": "⚠️ STAGE 2: Critical Phase Boundary Reached (N/d = 1.0)",
            "color": "#FFD700",
            "active_node": 1,
        },
        {
            "load": 1.8,
            "status": "🔴 STAGE 3: Softmax Exponential Noise Explosion! (N/d = 1.8)",
            "color": "#FF0055",
            "active_node": 2,
        },
        {
            "load": 2.2,
            "status": "💥 STAGE 4: Pythia-70M Deep Layer Rank Collapse Detected",
            "color": "#9900FF",
            "active_node": 3,
        },
        {
            "load": 1.0,
            "status": "⚡ STAGE 5: ETF-Aware Initialization Applied -> 100% Recovery!",
            "color": "#00E5FF",
            "active_node": 4,
        },
    ]

    for step in stages:
        status_box.markdown(f"### {step['status']}")

        # Build Neural Graph Figure
        fig = go.Figure()

        # Draw Network Connections (Edges)
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[2, 2, 2, 2, 2],
                mode="lines",
                line=dict(color="#444", width=4),
                hoverinfo="none",
            )
        )

        # Draw Neural Nodes
        colors = ["#333"] * 5
        colors[step["active_node"]] = step["color"]
        sizes = [30] * 5
        sizes[step["active_node"]] = 55

        fig.add_trace(
            go.Scatter(
                x=nodes_x,
                y=nodes_y,
                mode="markers+text",
                marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
                text=node_labels,
                textposition="top center",
                textfont=dict(size=14, color="white"),
            )
        )

        # Draw Dynamic Math Signal Wave below nodes
        x_wave = np.linspace(1, 5, 200)
        if step["active_node"] < 2:
            y_wave = 1 + 0.2 * np.sin(10 * x_wave)  # Low Noise
        elif step["active_node"] in [2, 3]:
            y_wave = 1 + 0.8 * np.random.randn(len(x_wave))  # Explosive Noise
        else:
            y_wave = 1 + 0.1 * np.sin(20 * x_wave)  # ETF Restored Signal

        fig.add_trace(
            go.Scatter(
                x=x_wave,
                y=y_wave,
                mode="lines",
                line=dict(color=step["color"], width=3),
                name="Attention Signal Integrity",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=480,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 3.5]),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        plot_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(3.5)  # Auto delay for 85-sec pitch timing
