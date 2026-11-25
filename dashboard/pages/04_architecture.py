"""
PongRL Pro — Model Architecture Deep Dive
Visual explanation of Nature CNN feature extractor and reinforcement learning heads.
"""
import os
import sys
import streamlit as st
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import NatureCNN, PolicyNetwork, QNetwork, ActorCriticNetwork, KarpathyPolicyNetwork
from dashboard.components.sidebar import render_sidebar
from dashboard.components.styles import inject_custom_styles
from dashboard.components.metrics import render_metric_card

st.set_page_config(
    page_title="Model Architecture — PongRL Pro",
    page_icon="🧠",
    layout="wide"
)

inject_custom_styles()


def main():
    render_sidebar()

    st.markdown("<div class='hero-title'>🧠 Neural Network Architectures</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Detailed structural specifications of the deep convolutional feature extractor and algorithm-specific network heads.</div>",
        unsafe_allow_html=True
    )

    tab_cnn, tab_karpathy, tab_interactive = st.tabs([
        "🏛️ Nature CNN (DQN / PPO / REINFORCE)",
        "⚡ Karpathy 2-Layer MLP (Baseline)",
        "🔬 Interactive Tensor Dimension Inspector"
    ])

    # ---------------- TAB 1: NATURE CNN ARCHITECTURE ----------------
    with tab_cnn:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">Nature DQN Convolutional Architecture (Mnih et al., 2015)</h4>
            <p style="color: #a7accc; margin: 0; line-height: 1.6;">
                The standard vision pipeline processes raw Atari frames into spatial feature representations.
                By stacking the 4 most recent grayscale 84×84 frames, the network observes temporal dynamics 
                (ball velocity and paddle motion) directly without recurrent layers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Mermaid Architecture Diagram
        st.markdown("### 📐 End-to-End Visual Architecture Diagram")
        
        st.markdown("""
        ```mermaid
        flowchart TD
            subgraph Preprocessing ["1. Preprocessing Pipeline"]
                RAW["Raw Atari Frame<br/>(210 x 160 x 3 RGB)"]
                CROP["Crop & Grayscale<br/>(84 x 84 x 1)"]
                STACK["Frame Stacking (k=4)<br/><b>Input: (4 x 84 x 84)</b>"]
                RAW --> CROP --> STACK
            end

            subgraph FeatureExtractor ["2. Nature CNN Feature Extractor"]
                CONV1["Conv2D (32 filters, 8x8, stride 4, ReLU)<br/>Output: (32 x 20 x 20)"]
                CONV2["Conv2D (64 filters, 4x4, stride 2, ReLU)<br/>Output: (64 x 9 x 9)"]
                CONV3["Conv2D (64 filters, 3x3, stride 1, ReLU)<br/>Output: (64 x 7 x 7)"]
                FLAT["Flatten Layer<br/>Output: (3136 features)"]
                FC["Dense Linear Layer (ReLU)<br/>Output: (512 latent vector)"]
                
                STACK --> CONV1 --> CONV2 --> CONV3 --> FLAT --> FC
            end

            subgraph Heads ["3. Task-Specific Output Heads"]
                FC --> HEAD_DQN["DQN Head: Linear(512, 6)<br/><b>Q(s, a) for all 6 actions</b>"]
                FC --> HEAD_REINFORCE["Policy Head: Linear(512, 6) + Softmax<br/><b>π(a | s) action probabilities</b>"]
                FC --> HEAD_PPO_PI["Actor Head: Linear(512, 6)<br/><b>Action Logits</b>"]
                FC --> HEAD_PPO_V["Critic Head: Linear(512, 1)<br/><b>State Value V(s)</b>"]
            end
        ```
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Layer Details Table
        st.markdown("### 📋 Layer-by-Layer Dimensions & Parameters")
        
        layers_data = [
            {"Stage": "Input Tensor", "Layer Type": "AtariObservation", "Input Shape": "(B, 4, 84, 84)", "Output Shape": "(B, 4, 84, 84)", "Params": "0", "Activation": "Normalized [0, 1]"},
            {"Stage": "Feature Extraction", "Layer Type": "Conv2d (32, 8x8, s=4)", "Input Shape": "(B, 4, 84, 84)", "Output Shape": "(B, 32, 20, 20)", "Params": "8,224", "Activation": "ReLU"},
            {"Stage": "Feature Extraction", "Layer Type": "Conv2d (64, 4x4, s=2)", "Input Shape": "(B, 32, 20, 20)", "Output Shape": "(B, 64, 9, 9)", "Params": "32,832", "Activation": "ReLU"},
            {"Stage": "Feature Extraction", "Layer Type": "Conv2d (64, 3x3, s=1)", "Input Shape": "(B, 64, 9, 9)", "Output Shape": "(B, 64, 7, 7)", "Params": "36,928", "Activation": "ReLU"},
            {"Stage": "Dense Representation", "Layer Type": "Flatten", "Input Shape": "(B, 64, 7, 7)", "Output Shape": "(B, 3136)", "Params": "0", "Activation": "None"},
            {"Stage": "Dense Representation", "Layer Type": "Linear (3136, 512)", "Input Shape": "(B, 3136)", "Output Shape": "(B, 512)", "Params": "1,606,144", "Activation": "ReLU"},
            {"Stage": "DQN Head", "Layer Type": "Linear (512, 6)", "Input Shape": "(B, 512)", "Output Shape": "(B, 6)", "Params": "3,078", "Activation": "Linear Q(s, a)"},
            {"Stage": "Policy Head", "Layer Type": "Linear (512, 6)", "Input Shape": "(B, 512)", "Output Shape": "(B, 6)", "Params": "3,078", "Activation": "Softmax π(a|s)"},
            {"Stage": "PPO Critic Head", "Layer Type": "Linear (512, 1)", "Input Shape": "(B, 512)", "Output Shape": "(B, 1)", "Params": "513", "Activation": "Scalar V(s)"},
        ]
        st.dataframe(pd.DataFrame(layers_data), use_container_width=True)

    # ---------------- TAB 2: KARPATHY MLP ARCHITECTURE ----------------
    with tab_karpathy:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">Karpathy 2-Layer Policy Network (REINFORCE Pong Baseline)</h4>
            <p style="color: #a7accc; margin: 0; line-height: 1.6;">
                The minimal Pong policy network introduced by Andrej Karpathy (2016). Instead of convolutional layers,
                it uses a downsampled 80×80 difference frame ($x_t - x_{t-1}$) flattened into a 6,400-dimensional vector.
                This eliminates static background elements and isolates movement.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c_k1, c_k2 = st.columns(2)
        with c_k1:
            st.markdown("""
            ```mermaid
            flowchart TD
                DIF["Difference Frame<br/>(x_t - x_{t-1})<br/><b>80 x 80 = 6,400 float</b>"]
                W1["W1 Weight Matrix (No Bias)<br/>Linear(6400, 200) + ReLU<br/><b>1,280,000 parameters</b>"]
                W2["W2 Weight Vector (No Bias)<br/>Linear(200, 1) + Sigmoid<br/><b>200 parameters</b>"]
                PROB["Output Action Probability<br/><b>p = σ(W2 · h)</b><br/>Probability of Action 2 (UP)"]
                ACT["Action Selection:<br/>Action 2 (UP) if p > 0.5<br/>Action 3 (DOWN) otherwise"]
                
                DIF --> W1 --> W2 --> PROB --> ACT
            ```
            """)

        with c_k2:
            st.markdown("""
            <div class="glass-card">
                <h4>Network Mathematical Specifications</h4>
                <ul>
                    <li><strong>Input Dimensionality</strong>: $D = 80 \\times 80 = 6,400$</li>
                    <li><strong>Hidden Layer Size</strong>: $H = 200$ neurons</li>
                    <li><strong>Activation</strong>: $\\text{ReLU}(h) = \\max(0, W_1 x)$</li>
                    <li><strong>Output Activation</strong>: $\\text{Sigmoid}(z) = \\frac{1}{1 + e^{-z}}$</li>
                    <li><strong>Total Weights</strong>: $1,280,200$ floats ($\\sim 5.12$ MB)</li>
                    <li><strong>Optimizer</strong>: Leaky RMSProp (decay $= 0.99$, $lr = 10^{-3}$)</li>
                    <li><strong>Checkpoint Loaded</strong>: <code>checkpoints/reinforce_karpathy_6000ep.pth</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- TAB 3: INTERACTIVE DIMENSION INSPECTOR ----------------
    with tab_interactive:
        st.markdown("### 🧪 Live PyTorch Tensor Forward Pass Verification")
        batch_size = st.slider("Select Batch Size for Forward Pass Simulation", min_value=1, max_value=64, value=4)

        if st.button("Run Forward Pass Simulation"):
            dummy_cnn_input = torch.randn(batch_size, 4, 84, 84)
            cnn = NatureCNN(in_channels=4)
            q_net = QNetwork(in_channels=4, num_actions=6)
            ppo_net = ActorCriticNetwork(in_channels=4, num_actions=6)

            with torch.no_grad():
                features = cnn(dummy_cnn_input)
                q_vals = q_net(dummy_cnn_input)
                logits, values = ppo_net(dummy_cnn_input)

            st.success("✅ Forward pass computed successfully through PyTorch dynamic graph!")
            
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.markdown(f"**Feature Vector:** `{tuple(features.shape)}`")
                st.caption("512-dimensional shared representation")
            with p_col2:
                st.markdown(f"**DQN Q-Values:** `{tuple(q_vals.shape)}`")
                st.caption("6 action Q-values per batch element")
            with p_col3:
                st.markdown(f"**PPO Actor / Critic:** `{tuple(logits.shape)}`, `{tuple(values.shape)}`")
                st.caption("Action logits + scalar state values V(s)")


if __name__ == "__main__":
    main()
