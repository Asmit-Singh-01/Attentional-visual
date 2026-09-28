import time
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Attention Phase Transition Simulator", layout="wide"
)

st.title("⚡ Geometric Phase Transition & Text Attention Collapse")

# Controls
st.sidebar.header("🕹️ Live Controller")
ratio = st.sidebar.slider("Feature Load Ratio (N / d_head)", 0.2, 2.5, 0.8, 0.1)
d_head = 16
N = int(d_head * ratio)

col1, col2 = st.columns(2)

# --- 1. Live Text Corruption Visualiser ---
with col1:
    st.subheader("1. Token Context Attention State")
    sample_text = "Transformers utilize multi-head self-attention to process sequential data."

    if ratio <= 1.0:
        st.success(f"🟢 [STATUS: STABLE] {sample_text}")
        st.info(f"Capacity Load: {ratio:.2f} <= 1.0 (Zero Cross-Talk Noise)")
    else:
        # Corrupt text based on overload ratio
        corrupted = "".join(
            [
                (
                    char
                    if np.random.rand() > (ratio - 1.0) / 2.0
                    else np.random.choice(["█", "░", "#", "!", "⚡", "?"])
                )
                for char in sample_text
            ]
        )
        st.error(f"🔴 [STATUS: COLLAPSED] {corrupted}")
        st.warning(
            f"Cross-Talk Noise Amplified! N/d_head = {ratio:.2f} > 1.0 (Phase Transition Exceeded)"
        )

# --- 2. Live Softmax Cross-Talk Heatmap ---
with col2:
    st.subheader("2. Softmax Exponential Cross-Talk Noise")

    # Generate Query-Key Score Matrix
    np.random.seed(42)
    Q = np.random.randn(N, d_head)
    K = np.random.randn(N, d_head)
    S = np.dot(Q, K.T) / np.sqrt(d_head)

    # Softmax
    S_soft = np.exp(S - np.max(S, axis=-1, keepdims=True))
    S_soft /= np.sum(S_soft, axis=-1, keepdims=True)

    fig = px.imshow(
        S_soft,
        color_continuous_scale="Reds" if ratio > 1.0 else "Greens",
        title=f"Post-Softmax Attention Matrix (Load = {ratio:.2f})",
    )
    fig.update_layout(
        template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
