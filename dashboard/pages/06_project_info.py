"""
PongRL Pro — Project Documentation & Academic Review Guide
Comprehensive capstone project overview for guide review and viva examination.
"""
import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.components.sidebar import render_sidebar
from dashboard.components.styles import inject_custom_styles

st.set_page_config(
    page_title="Project Information — PongRL Pro",
    page_icon="📑",
    layout="wide"
)

inject_custom_styles()


def main():
    render_sidebar()

    st.markdown("<div class='hero-title'>📑 Project Documentation & Review Dossier</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Academic defense reference, problem statement, mathematical foundations, and system architecture.</div>",
        unsafe_allow_html=True
    )

    tab_overview, tab_theory, tab_architecture, tab_tech, tab_viva = st.tabs([
        "🎯 Problem & Objectives",
        "🧮 Mathematical RL Theory",
        "🏛️ System Architecture",
        "🛠️ Technologies & Tools",
        "🎓 Viva & Review FAQ"
    ])

    # ---------------- TAB 1: PROBLEM & OBJECTIVES ----------------
    with tab_overview:
        st.markdown("""
        ### 1. Problem Statement
        Classic control and reinforcement learning algorithms historically operated on low-dimensional internal state vectors
        (e.g., exact paddle coordinates, ball velocities). However, autonomous real-world systems must perceive and act
        directly from raw sensory observations.

        The problem addressed in **PongRL Pro** is:
        > *Can an artificial agent learn superhuman control policies in the classic Atari 2600 game Pong exclusively
        > from raw visual pixel streams, without hand-crafted heuristics or human gameplay rules?*

        ### 2. Core Project Objectives
        1. **End-to-End Perception and Control**: Transform $210 \\times 160 \\times 3$ RGB pixel buffers into discrete paddle actions through deep neural networks.
        2. **Algorithm Implementation**: Implement and benchmark three distinct deep reinforcement learning paradigms natively:
           - Policy Gradient (REINFORCE)
           - Value-Based Q-Learning (Double DQN)
           - Actor-Critic (Proximal Policy Optimization - PPO)
        3. **Rigorous Experimental Validation**: Demonstrate empirical convergence through recorded training runs (6,000 episodes) and verify policy progression.
        4. **Interactive AI Dashboard**: Provide a modern research dashboard for real-time inference, model evaluation, and parameter tuning.
        """)

    # ---------------- TAB 2: MATHEMATICAL THEORY ----------------
    with tab_theory:
        st.markdown("""
        ### Mathematical Reinforcement Learning Foundations

        #### 1. Markov Decision Process (MDP)
        The visual Atari environment is formalized as a partially observable Markov Decision Process (POMDP) transformed into an MDP via frame stacking:
        $$\\mathcal{M} = \\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma \\rangle$$
        - $\\mathcal{S}$: State space consisting of 4 stacked $84 \\times 84$ grayscale frames.
        - $\\mathcal{A}$: Discrete action space containing 6 valid actions $\\{0: \\text{NOOP}, 1: \\text{FIRE}, 2: \\text{UP}, 3: \\text{DOWN}, 4: \\text{UPFIRE}, 5: \\text{DOWNFIRE}\\}$.
        - $\\mathcal{P}(s' | s, a)$: State transition probability modeled by the Stella Atari 2600 emulator.
        - $\\mathcal{R}(s, a)$: Reward signal ($+1$ for scoring a point, $-1$ for conceding a point, $0$ during rallies).
        - $\\gamma = 0.99$: Discount factor prioritizing future rewards over immediate actions.

        #### 2. Bellman Optimality Equation (DQN)
        The optimal action-value function satisfies the Bellman recurrence relation:
        $$Q^*(s, a) = \\mathcal{R}(s, a) + \\gamma \\sum_{s'} \\mathcal{P}(s' | s, a) \\max_{a'} Q^*(s', a')$$

        In **Double DQN**, overestimation bias is eliminated by decoupling selection and evaluation:
        $$y_t = r_t + \\gamma Q(s_{t+1}, \\arg\\max_{a} Q(s_{t+1}, a; \\theta); \\theta^-)$$

        #### 3. Policy Gradient Theorem (REINFORCE)
        Policy parameters are updated via stochastic gradient ascent on the expected cumulative reward $J(\\theta)$:
        $$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\pi_\\theta} \\left[ \\sum_{t=0}^T \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\cdot (G_t - b) \\right]$$
        where $G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k}$ is the discounted return, and $b$ is a variance-reduction baseline.
        """)

    # ---------------- TAB 3: SYSTEM ARCHITECTURE ----------------
    with tab_architecture:
        st.markdown("""
        ### Modular Software Engineering Architecture
        PongRL Pro is architected with strict object-oriented separation of concerns:

        ```mermaid
        flowchart LR
            subgraph Config ["Configuration"]
                CFG["Config Manager<br/>(src/config.py)"]
            end

            subgraph Environment ["Environment Layer"]
                ALE["Gymnasium ALE<br/>(ALE/Pong-v5)"]
                WRAP["AtariPreprocessing &<br/>FrameStackObservation<br/>(src/env_wrapper.py)"]
                ALE --> WRAP
            end

            subgraph AgentCore ["Agent & Intelligence"]
                MODELS["Neural Networks<br/>Nature CNN / MLP<br/>(src/model.py)"]
                AGENTS["RL Agents<br/>DQN / PPO / REINFORCE<br/>(src/agent.py)"]
                BUFFERS["Replay / Rollout Buffers<br/>(src/replay.py)"]
                AGENTS --> MODELS
                AGENTS --> BUFFERS
            end

            subgraph Orchestration ["Pipelines"]
                TRAINER["Trainer Loop<br/>(src/trainer.py)"]
                EVAL["Evaluator<br/>(src/evaluator.py)"]
                SERVICE["Backend Service<br/>(src/service.py)"]
            end

            subgraph UI ["User Interface"]
                DASH["Streamlit Research Dashboard<br/>(dashboard/app.py)"]
            end

            CFG --> TRAINER
            CFG --> EVAL
            WRAP --> TRAINER
            WRAP --> EVAL
            AGENTS --> TRAINER
            AGENTS --> EVAL
            SERVICE --> DASH
            AGENTS --> SERVICE
        ```
        """)

    # ---------------- TAB 4: TECHNOLOGIES ----------------
    with tab_tech:
        st.markdown("""
        ### Technology Stack & Dependency Justification

        | Component | Technology | Role & Justification |
        |-----------|------------|----------------------|
        | **Deep Learning Framework** | `PyTorch 2.x` | Native dynamic computation graph, CUDA GPU acceleration, auto-differentiation. |
        | **RL Environment** | `Gymnasium 1.0+` | Official maintained Farama Foundation standard successor to OpenAI Gym. |
        | **Atari Emulation** | `ale-py 0.12+` | Arcade Learning Environment Stella emulator with zero licensing obstacles. |
        | **Frontend Dashboard** | `Streamlit 1.30+` | Real-time reactive web application with interactive widgets and state persistence. |
        | **Interactive Visualizations** | `Plotly 5.18+` | Hardware-accelerated WebGL charts, dark theme templates, multi-trace telemetry. |
        | **Numerical Computing** | `NumPy 2.x`, `Pandas 2.x` | High-throughput tensor reshaping, moving averages, and trajectory calculations. |
        | **Experiment Logging** | `TensorBoard 2.x` | Scalable scalar metric logging, loss tracking, and learning curves. |
        """)

    # ---------------- TAB 5: VIVA FAQ ----------------
    with tab_viva:
        st.markdown("""
        ### Key Project Guide Defense Questions & Answers

        **Q1: Why is frame stacking necessary for Atari Pong?**  
        *Answer:* A single static 2D frame does not convey the velocity or trajectory of the ball or paddle. By stacking 4 consecutive frames $(4 \\times 84 \\times 84)$, the temporal first and second derivatives (velocity and acceleration) are encoded spatially, transforming the non-Markovian single frame into a Markovian state.

        **Q2: What is the purpose of the Target Network in DQN?**  
        *Answer:* In standard Q-learning with neural networks, the target value $r + \\gamma \\max Q(s', a')$ changes with every single gradient step because the same network computes both the prediction and the target. This causes oscillations and instability. The Target Network freezes target weights for $1,000$ steps, stabilizing training.

        **Q3: How was the 6,000-episode baseline trained?**  
        *Answer:* It used the REINFORCE policy gradient algorithm with baseline standardization and RMSProp optimization. The agent started at $-20.4$ average reward (losing 21-0) and steadily improved over 6,000 episodes to reach a peak running reward of $-1.47$, demonstrating skilled rallies and defensive mastery.

        **Q4: Can legacy TensorFlow model weights be loaded directly into PyTorch?**  
        *Answer:* No. TensorFlow checkpoint files (such as `ModelWeights.index` and `ModelWeights.data`) use Google Protobuf and TF Variable format, which is structurally incompatible with PyTorch's `torch.load` state dictionary format. Our project clearly identifies legacy TF checkpoints as incompatible and provides native PyTorch `.pth` models and adapters.
        """)


if __name__ == "__main__":
    main()

