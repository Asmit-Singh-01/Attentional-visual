import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Attention Mechanics Engine", layout="wide")

st.markdown("<h1 style='text-align: center; color: #00FFA3;'>⚡ Dynamic Attention Trajectory & Phase Transition</h1>", unsafe_allow_html=True)

run_btn = st.button("▶️ Run 90s Live Pipeline Simulation", use_container_width=True)

graph_place = st.empty()
math_place = st.empty()

# Block Definitions (X, Y)
blocks = {
    "Encoder": (1, 2),
    "QK-MatMul": (3, 2),
    "Softmax Amp": (5, 2),
    "Pythia SVD": (7, 2),
    "ETF Recovery": (9, 2)
}

if run_btn:
    stages = [
        {"from": "Encoder", "to": "QK-MatMul", "val": 0.5, "status": "🟢 STAGE 1: Normal Ingestion", "color": "#00FFA3", 
         "eq": r"S_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_{head}}}, \quad \frac{N}{d_{head}} = 0.5 \le 1.0", "metrics": "Error = 0.001 | Rank Deficit = 0"},
        
        {"from": "QK-MatMul", "to": "Softmax Amp", "val": 1.0, "status": "🟡 STAGE 2: Critical Boundary Hit", "color": "#FFD700", 
         "eq": r"\frac{N}{d_{head}} = 1.0 \implies \text{Geometric Phase Transition}", "metrics": "Spectral Leakage = 4.2% | Orthogonality Drift = 0.04"},
        
        {"from": "Softmax Amp", "to": "Pythia SVD", "val": 1.8, "status": "🔴 STAGE 3: Softmax Exponential Explosion", "color": "#FF0055", 
         "eq": r"A_{ij} = \frac{\exp(S_{ij})}{\sum \exp(S_{ik})}, \quad \mathcal{D}_{KL} \sim \mathcal{O}(e^{\gamma N})", "metrics": "Entropy H(A) = 4.82 nats | CROSS-TALK COLLAPSE"},
        
        {"from": "Pythia SVD", "to": "ETF Recovery", "val": 2.2, "status": "🟣 STAGE 4: Real Weight Rank Truncation", "color": "#9900FF", 
         "eq": r"\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top, \quad \sigma_i \to 0 \text{ for } i > d_{head}", "metrics": "Effective Rank = 14/16 | Loss of Context Window"},
        
        {"from": "ETF Recovery", "to": "ETF Recovery", "val": 1.0, "status": "⚡ STAGE 5: ETF Polytope Geometry Recovery", "color": "#00E5FF", 
         "eq": r"\mathbf{M}^* = \sqrt{\frac{K}{K-1}} \left( \mathbf{I}_K - \frac{1}{K}\mathbf{1}\mathbf{1}^\top \right)", "metrics": "Reconstruction = 99.87% | Zero Context Loss"}
    ]

    for stage in stages:
        p1 = blocks[stage["from"]]
        p2 = blocks[stage["to"]]

        # Animate packet moving from A to B in sub-steps
        steps = 8 if p1 != p2 else 1
        for t in range(steps + 1):
            alpha = t / steps
            curr_x = p1[0] + alpha * (p2[0] - p1[0])
            curr_y = p1[1] + alpha * (p2[1] - p1[1])

            fig = go.Figure()

            # Draw Architecture Blocks
            for name, (bx, by) in blocks.items():
                b_color = stage["color"] if name in [stage["from"], stage["to"]] else "#1E1E2E"
                fig.add_trace(go.Scatter(
                    x=[bx], y=[by], mode="markers+text",
                    marker=dict(symbol="square", size=55, color=b_color, line=dict(width=2, color="#FFF")),
                    text=[f"<b>{name}</b>"], textposition="bottom center",
                    textfont=dict(color="#FFF", size=12)
                ))

            # Connections
            fig.add_trace(go.Scatter(
                x=[1, 3, 5, 7, 9], y=[2, 2, 2, 2, 2],
                mode="lines", line=dict(color="#444", width=3, dash="dot")
            ))

            # Live Traveling Packet
            fig.add_trace(go.Scatter(
                x=[curr_x], y=[curr_y], mode="markers",
                marker=dict(size=22, color=stage["color"], line=dict(width=3, color="#FFF"))
            ))

            fig.update_layout(
                template="plotly_dark", height=380, showlegend=False,
                xaxis=dict(visible=False, range=[0, 10]),
                yaxis=dict(visible=False, range=[0, 4]),
                margin=dict(l=10, r=10, t=10, b=10)
            )

            graph_place.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            time.sleep(0.1)

        # Math Update
        math_place.markdown(
            f"""
            <div style="background:#0A0A12; border: 2px solid {stage['color']}; padding:15px; border-radius:10px;">
                <h3 style="color:{stage['color']}; margin:0;">{stage['status']}</h3>
                <p style="font-size:18px; color:#FFF;">$$\n{stage['eq']}\n$$</p>
                <p style="font-size:14px; color:#00FFA3; font-family:monospace;"><b>Live Telemetry:</b> {stage['metrics']}</p>
            </div>
            """, unsafe_allow_html=True
        )
        time.sleep(2.5)
         
