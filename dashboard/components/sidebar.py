"""
Sidebar component for PongRL Pro Dashboard.
"""
import streamlit as st
import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_sidebar():
    """Renders the standard sidebar with project info, system status, and links."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 2.4rem; line-height: 1.2;">🏓</div>
            <h2 style="
                background: linear-gradient(135deg, #00d4ff, #7c3aed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 6px 0 2px 0;
                font-weight: 800;
                letter-spacing: -0.5px;
            ">PongRL Pro</h2>
            <div style="color: #8e92a8; font-size: 0.78rem; font-weight: 500; line-height: 1.3;">
                Deep Reinforcement Learning Atari Agent
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # System Hardware & Runtime Status
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        device_color = "#00ffaa" if torch.cuda.is_available() else "#00d4ff"
        
        st.markdown("#### ⚡ System Runtime")
        st.markdown(f"""
        <div class="sidebar-info-box">
            <div class="sidebar-info-row">
                <span class="info-dim">PyTorch:</span>
                <span class="info-val">{torch.__version__.split('+')[0]}</span>
            </div>
            <div class="sidebar-info-row">
                <span class="info-dim">Device:</span>
                <span class="info-val" style="color: {device_color}; font-weight: 600;">{device_name}</span>
            </div>
            <div class="sidebar-info-row">
                <span class="info-dim">Env ID:</span>
                <span class="info-val">ALE/Pong-v5</span>
            </div>
            <div class="sidebar-info-row">
                <span class="info-dim">Action Space:</span>
                <span class="info-val">Discrete(6)</span>
            </div>
            <div class="sidebar-info-row">
                <span class="info-dim">Observation:</span>
                <span class="info-val">(4, 84, 84) CNN</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # Quick Documentation Links
        st.markdown("#### 📖 Project Guide")
        st.markdown("""
        - **Algorithms**: REINFORCE, DQN, PPO
        - **Environment**: Gymnasium Arcade ALE
        - **Feature Extractor**: Nature CNN (Mnih 2015)
        - **Trained Baseline**: 6,000-Ep Policy Gradient
        """)

        # Footer
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #555b77; font-size: 0.75rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 15px;">
            Academic Capstone RL Project<br>
            Developed for Guide Review & Viva
        </div>
        """, unsafe_allow_html=True)
