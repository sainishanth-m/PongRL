"""
PongRL Pro — Hyperparameter Configuration Manager
Interactive configuration builder that persists parameters directly to disk.
"""
import os
import sys
import yaml
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.service import BackendService
from src.config import Config
from dashboard.components.sidebar import render_sidebar
from dashboard.components.styles import inject_custom_styles
from dashboard.components.metrics import render_status_pill

st.set_page_config(
    page_title="Hyperparameters — PongRL Pro",
    page_icon="🎛️",
    layout="wide"
)

inject_custom_styles()


def main():
    render_sidebar()

    st.markdown("<div class='hero-title'>🎛️ Hyperparameter Configuration</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Tune reinforcement learning hyperparameters and persist configurations directly to valid YAML files for training pipelines.</div>",
        unsafe_allow_html=True
    )

    algo = st.selectbox("Select Target Algorithm to Configure", ["PPO", "DQN", "REINFORCE"])
    
    # Load existing algorithm YAML configuration
    yaml_filename = f"{algo.lower()}.yaml"
    existing_yaml_path = os.path.join(PROJECT_ROOT, "config", yaml_filename)
    
    curr_data = {}
    if os.path.exists(existing_yaml_path):
        with open(existing_yaml_path, "r") as f:
            curr_data = yaml.safe_load(f) or {}

    curr_training = curr_data.get("training", {})

    col_tune, col_preview = st.columns([3, 2])

    with col_tune:
        st.markdown(f"### ⚙️ Tune {algo} Hyperparameters")

        # Global Training Settings
        st.markdown("#### Common Parameters")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            lr = st.select_slider(
                "Learning Rate (Adam / RMSProp)",
                options=[1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3],
                value=float(curr_training.get("learning_rate", 2.5e-4)),
                format_func=lambda x: f"{x:.1e}"
            )
            gamma = st.slider(
                "Discount Factor (Gamma $\\gamma$)",
                min_value=0.85, max_value=0.999,
                value=float(curr_training.get("gamma", 0.99)),
                step=0.005
            )

        with g_col2:
            batch_size = st.select_slider(
                "Batch Size",
                options=[1, 16, 32, 64, 128, 256],
                value=int(curr_training.get("batch_size", 64 if algo == "PPO" else 32))
            )
            total_steps = st.select_slider(
                "Total Environment Steps",
                options=[50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                value=int(curr_training.get("total_timesteps", 1_000_000)),
                format_func=lambda x: f"{x:,} steps"
            )

        # Algorithm-Specific Settings
        st.markdown(f"#### {algo}-Specific Hyperparameters")
        algo_specific = {}

        if algo == "PPO":
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                n_steps = st.slider("Rollout Steps (n_steps)", 32, 512, int(curr_training.get("n_steps", 128)), step=32)
                clip_range = st.slider("PPO Clipping Range ($\\epsilon$)", 0.05, 0.4, float(curr_training.get("clip_range", 0.1)), step=0.05)
                gae_lambda = st.slider("GAE Lambda ($\\lambda$)", 0.8, 1.0, float(curr_training.get("gae_lambda", 0.95)), step=0.01)
            with p_col2:
                n_epochs = st.slider("Update Epochs (n_epochs)", 1, 10, int(curr_training.get("n_epochs", 4)))
                ent_coef = st.slider("Entropy Coefficient ($c_{ent}$)", 0.0, 0.05, float(curr_training.get("ent_coef", 0.01)), step=0.005)
                vf_coef = st.slider("Value Function Coefficient ($c_{vf}$)", 0.1, 1.0, float(curr_training.get("vf_coef", 0.5)), step=0.1)
                
            algo_specific = {
                "n_steps": n_steps,
                "clip_range": clip_range,
                "gae_lambda": gae_lambda,
                "n_epochs": n_epochs,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef
            }

        elif algo == "DQN":
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                buffer_size = st.select_slider("Replay Buffer Size", options=[10_000, 50_000, 100_000, 200_000], value=int(curr_training.get("buffer_size", 100_000)))
                target_update = st.slider("Target Network Update Freq", 100, 5000, int(curr_training.get("target_update_freq", 1000)), step=100)
                learning_starts = st.slider("Learning Starts (Warmup Steps)", 1000, 20000, int(curr_training.get("learning_starts", 10000)), step=1000)
            with d_col2:
                eps_init = st.slider("Initial Epsilon", 0.5, 1.0, float(curr_training.get("exploration_initial_eps", 1.0)), step=0.05)
                eps_final = st.slider("Final Epsilon", 0.001, 0.1, float(curr_training.get("exploration_final_eps", 0.01)), step=0.005)
                eps_frac = st.slider("Exploration Fraction", 0.05, 0.5, float(curr_training.get("exploration_fraction", 0.1)), step=0.05)

            algo_specific = {
                "buffer_size": buffer_size,
                "target_update_freq": target_update,
                "learning_starts": learning_starts,
                "exploration_initial_eps": eps_init,
                "exploration_final_eps": eps_final,
                "exploration_fraction": eps_frac
            }

        elif algo == "REINFORCE":
            entropy_coef = st.slider("Entropy Regularization Coefficient", 0.0, 0.05, float(curr_training.get("entropy_coef", 0.01)), step=0.005)
            algo_specific = {
                "entropy_coef": entropy_coef
            }

        # Build payload
        payload = {
            "algorithm": algo.lower(),
            "training": {
                "learning_rate": lr,
                "gamma": gamma,
                "batch_size": batch_size,
                "total_timesteps": total_steps,
                **algo_specific
            },
            "model": {
                "hidden_size": 512
            }
        }

        # File target selection
        dest_choice = st.radio(
            "Save Target",
            [f"Standard ({yaml_filename})", f"Custom (config/custom_{algo.lower()}.yaml)"],
            horizontal=True
        )
        if "Custom" in dest_choice:
            save_path = os.path.join(PROJECT_ROOT, "config", f"custom_{algo.lower()}.yaml")
        else:
            save_path = existing_yaml_path

        # Save Button
        if st.button(f"💾 Save {algo} Configuration to Disk", type="primary"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
            st.success(f"✅ Configuration successfully saved to `{save_path}`!")
            st.toast(f"Configuration written to {os.path.basename(save_path)}", icon="💾")

    # Right Column: Live YAML Preview & Execution Snippet
    with col_preview:
        st.markdown("### 📄 Real-Time YAML Preview")
        preview_yaml = yaml.dump(payload, default_flow_style=False, sort_keys=False)
        st.code(preview_yaml, language="yaml")

        st.markdown("""
        <div class="glass-card">
            <h4>🚀 Run Training with Saved Config</h4>
            <p style="color: #a7accc; font-size: 0.85rem;">
                Launch the PyTorch training pipeline directly from your terminal using the generated configuration:
            </p>
        """, unsafe_allow_html=True)
        
        cmd = f"python train.py --config config/default.yaml --algo-config {os.path.relpath(save_path, PROJECT_ROOT)} --algorithm {algo.lower()} --experiment-name {algo.lower()}_run"
        st.code(cmd, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
