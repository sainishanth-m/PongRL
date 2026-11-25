"""
PongRL Pro — Command Center & Overview Dashboard
Intelligent Game Agent Using Deep Reinforcement Learning
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is accessible
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.service import BackendService
from dashboard.components.sidebar import render_sidebar
from dashboard.components.styles import inject_custom_styles
from dashboard.components.metrics import render_metric_card, render_status_pill
from dashboard.components.charts import plot_real_reward_curve, plot_algorithm_comparison_bar

# Streamlit Page Configuration (SINGLE call, must be first Streamlit command)
st.set_page_config(
    page_title="PongRL Pro — Overview Command Center",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply unified CSS styles
inject_custom_styles()


def main():
    render_sidebar()
    
    # Hero Title Banner
    st.markdown("<div class='hero-title'>PongRL Pro Command Center</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Intelligent Game Agent Using Deep Reinforcement Learning & Computer Vision</div>",
        unsafe_allow_html=True
    )

    # Load actual project data
    real_data = BackendService.load_real_training_data()
    checkpoints = BackendService.list_checkpoints()
    recordings = BackendService.list_gameplay_recordings()

    # Introduction Card
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <h3 style="margin-top: 0; margin-bottom: 8px;">Executive Summary</h3>
                <p style="color: #a7accc; margin: 0; max-width: 850px; line-height: 1.6;">
                    <strong>PongRL Pro</strong> implements and benchmarks deep reinforcement learning agents on the Gymnasium Atari Pong environment.
                    Learning exclusively from raw pixels without handcrafted game rules, agents extract visual dynamics via a 3-layer Nature Convolutional Neural Network
                    and optimize policy and value networks through policy gradient and Q-learning formulations.
                </p>
            </div>
            <div>
                {render_status_pill("Baseline: 6,000 Ep Completed", "success")}
                {render_status_pill("PyTorch Ready", "info")}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Actual Performance Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        total_ep_str = f"{real_data['total_episodes']:,}" if real_data['total_episodes'] > 0 else "N/A"
        render_metric_card(
            label="Recorded Episodes",
            value=total_ep_str,
            delta="+6,001 Total",
            icon="🔄",
            subtitle="Full Empirical Baseline Run"
        )
        
    with m_col2:
        best_rew = real_data['best_reward']
        render_metric_card(
            label="Peak Running Reward",
            value=f"{best_rew:+.2f}",
            delta="+18.95 Gain",
            icon="🏆",
            subtitle="Convergence near 0 score boundary"
        )
        
    with m_col3:
        initial_rew = real_data['initial_reward']
        render_metric_card(
            label="Initial Baseline Reward",
            value=f"{initial_rew:+.1f}",
            delta="-20.4 at Ep 1",
            delta_color="inverse",
            icon="📉",
            subtitle="Untrained Random Play"
        )
        
    with m_col4:
        compat_ckpts = [c for c in checkpoints if c['compatible']]
        render_metric_card(
            label="Active Models",
            value=str(len(compat_ckpts)),
            delta=f"{len(checkpoints)} Discovered",
            icon="🧠",
            subtitle="PyTorch & Karpathy Checkpoints"
        )

    # Real Training Reward Progression Chart
    st.markdown("### 📈 Empirical Training Convergence")
    if real_data["has_data"] and not real_data["df"].empty:
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_reward = plot_real_reward_curve(
                real_data["df"],
                title=f"Actual Training Reward Evolution ({real_data['total_episodes']:,} Episodes)"
            )
            st.plotly_chart(fig_reward, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔬 Run Details")
            st.markdown(f"""
            - **Algorithm**: `REINFORCE`
            - **Environment**: `ALE/Pong-v5`
            - **Optimizer**: RMSProp ($lr = 10^{-3}$)
            - **Discount ($\\gamma$)**: `0.99`
            - **Input**: 80×80 Difference Frame
            - **Log Source**:  
              `{real_data['log_source']}`
            """)
            st.markdown("---")
            st.markdown("#### 🎮 Action Space")
            st.markdown("""
            - `Action 2`: Right / Paddle UP
            - `Action 3`: Left / Paddle DOWN
            - `Action 0`: NOOP (Idle)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No training logs found. Start training using `python train.py` to record live metrics.")

    # Checkpoint Inventory & Compatibility Audit
    st.markdown("### 💾 Checkpoint Inventory & Backend Compatibility")
    if checkpoints:
        ckpt_rows = []
        for c in checkpoints:
            status_badge = "✅ Compatible" if c["compatible"] else "❌ Incompatible (TensorFlow)"
            ckpt_rows.append({
                "Checkpoint Name": c["name"],
                "Format": c["type"],
                "Size (MB)": c["size_mb"],
                "Algorithm": c["algorithm"],
                "Compatibility": status_badge,
                "Backend Status": c["status"]
            })
        st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True)
    else:
        st.warning("No checkpoint files found in project.")

    # Project Presentation & Documentation Section
    st.markdown("### 📑 Project Presentation & Review Materials")
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.markdown("""
        <div class="glass-card">
            <h4>College Project Guide Demonstration Kit</h4>
            <p style="color: #a7accc; margin-bottom: 15px;">
                Download the complete academic capstone presentation slide deck covering system design, 
                Markov Decision Process (MDP) formulation, Bellman equations, and comparative analysis.
            </p>
        """, unsafe_allow_html=True)
        
        ppt_path = os.path.join(PROJECT_ROOT, "presentation.pptx")
        if os.path.exists(ppt_path):
            with open(ppt_path, "rb") as f:
                ppt_bytes = f.read()
            st.download_button(
                label="📥 Download Presentation Deck (presentation.pptx)",
                data=ppt_bytes,
                file_name="PongRL_Pro_Academic_Presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary"
            )
        else:
            st.info("Presentation PPT file can be generated with `python scripts/generate_presentation.py`.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_p2:
        st.markdown("""
        <div class="glass-card">
            <h4>Quick Navigation</h4>
            <ul style="color: #00d4ff; padding-left: 20px; line-height: 1.8;">
                <li><strong>Live Pong Gameplay</strong>: Interactive agent demo</li>
                <li><strong>Training Monitor</strong>: Metrics & reward curves</li>
                <li><strong>Algorithm Comparison</strong>: REINFORCE vs DQN vs PPO</li>
                <li><strong>Model Architecture</strong>: Deep CNN diagram</li>
                <li><strong>Hyperparameter Tuning</strong>: Interactive YAML editor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
