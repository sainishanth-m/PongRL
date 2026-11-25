"""
Backend Service Layer for PongRL Pro.

Provides a unified interface between the reinforcement learning backend
(agents, environments, models, loggers) and frontend dashboard interfaces.
"""
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.agent import BaseAgent, DQNAgent, ReinforceAgent, PPOAgent, KarpathyPolicyAgent
from src.model import NatureCNN, SimpleMLP, KarpathyPolicyNetwork
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class BackendService:
    """Centralized service managing checkpoints, logs, configs, and live inference."""
    
    ACTION_NAMES = {
        0: "NOOP (Idle)",
        1: "FIRE (Serve Ball)",
        2: "RIGHT / UP (Move Paddle Up)",
        3: "LEFT / DOWN (Move Paddle Down)",
        4: "RIGHTFIRE (Up & Serve)",
        5: "LEFTFIRE (Down & Serve)"
    }

    @staticmethod
    def get_project_root() -> str:
        return PROJECT_ROOT

    @classmethod
    def load_real_training_data(cls) -> Dict[str, Any]:
        """
        Loads actual recorded training logs from legacy_baseline/logs and runs/.
        Never returns fabricated numbers.
        """
        log_dir = os.path.join(PROJECT_ROOT, "legacy_baseline", "logs")
        primary_log = os.path.join(log_dir, "ControlRRLR1e-3.txt")
        baseline_log = os.path.join(log_dir, "ControlRunningRewards.txt")
        
        runs_dir = os.path.join(PROJECT_ROOT, "runs")
        
        results = {
            "has_data": False,
            "episodes": [],
            "running_rewards": [],
            "df": pd.DataFrame(),
            "best_reward": -21.0,
            "initial_reward": -21.0,
            "final_reward": -21.0,
            "total_episodes": 0,
            "algorithm": "REINFORCE (Policy Gradient)",
            "log_source": ""
        }

        # 1. Check for real 6001-episode training logs
        if os.path.exists(primary_log):
            try:
                with open(primary_log, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0:
                    episodes = list(range(len(data)))
                    rewards = [float(r) for r in data]
                    df = pd.DataFrame({
                        "episode": episodes,
                        "running_reward": rewards,
                        "moving_avg_50": pd.Series(rewards).rolling(50, min_periods=1).mean().values,
                        "moving_avg_100": pd.Series(rewards).rolling(100, min_periods=1).mean().values,
                    })
                    results["has_data"] = True
                    results["episodes"] = episodes
                    results["running_rewards"] = rewards
                    results["df"] = df
                    results["best_reward"] = float(max(rewards))
                    results["initial_reward"] = float(rewards[0])
                    results["final_reward"] = float(rewards[-1])
                    results["total_episodes"] = len(rewards)
                    results["log_source"] = "legacy_baseline/logs/ControlRRLR1e-3.txt"
                    return results
            except Exception as e:
                print(f"Error loading {primary_log}: {e}")

        # 2. Check for runs/ experiment metrics
        if os.path.exists(runs_dir):
            for exp in sorted(os.listdir(runs_dir), reverse=True):
                metrics_file = os.path.join(runs_dir, exp, "metrics.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r") as f:
                            m_data = json.load(f)
                        if m_data:
                            df = pd.DataFrame(m_data)
                            results["has_data"] = True
                            results["df"] = df
                            results["log_source"] = f"runs/{exp}/metrics.json"
                            results["total_episodes"] = len(df)
                            if "episode_reward" in df.columns:
                                results["best_reward"] = float(df["episode_reward"].max())
                                results["initial_reward"] = float(df["episode_reward"].iloc[0])
                                results["final_reward"] = float(df["episode_reward"].iloc[-1])
                            return results
                    except Exception:
                        pass

        return results

    @classmethod
    def list_checkpoints(cls) -> List[Dict[str, Any]]:
        """
        Scans all available checkpoints in the project and determines compatibility.
        """
        checkpoints = []
        
        # 1. Project checkpoints
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".pth") or f.endswith(".pt"):
                    p = os.path.join(ckpt_dir, f)
                    checkpoints.append({
                        "name": f,
                        "path": p,
                        "type": "PyTorch",
                        "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                        "compatible": True,
                        "algorithm": "REINFORCE" if "reinforce" in f else "PPO/DQN",
                        "status": "Ready for Live Gameplay & Evaluation"
                    })

        # 2. Legacy baseline checkpoints
        legacy_ckpt_dir = os.path.join(PROJECT_ROOT, "legacy_baseline", "checkpoints")
        if os.path.exists(legacy_ckpt_dir):
            for f in os.listdir(legacy_ckpt_dir):
                p = os.path.join(legacy_ckpt_dir, f)
                if f.endswith(".p") or f.endswith(".pkl"):
                    checkpoints.append({
                        "name": f"legacy/{f}",
                        "path": p,
                        "type": "Pickle (NumPy Weights)",
                        "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                        "compatible": True,
                        "algorithm": "REINFORCE (Karpathy Policy Gradient)",
                        "status": "Directly Playable via KarpathyPolicyAgent adapter"
                    })
                elif "ModelWeights" in f and f.endswith(".index"):
                    checkpoints.append({
                        "name": "legacy/ModelWeights (TensorFlow)",
                        "path": p,
                        "type": "TensorFlow 2 Checkpoint",
                        "size_mb": round(os.path.getsize(p) / 1024, 1),
                        "compatible": False,
                        "algorithm": "DQN (TensorFlow Experimental)",
                        "status": "Incompatible with PyTorch backend (legacy TF2 SavedModel format)"
                    })

        # 3. Check for runs/ checkpoints
        runs_dir = os.path.join(PROJECT_ROOT, "runs")
        if os.path.exists(runs_dir):
            for exp in os.listdir(runs_dir):
                exp_ckpts = os.path.join(runs_dir, exp, "checkpoints")
                if os.path.exists(exp_ckpts):
                    for f in os.listdir(exp_ckpts):
                        if f.endswith(".pth"):
                            p = os.path.join(exp_ckpts, f)
                            checkpoints.append({
                                "name": f"runs/{exp}/{f}",
                                "path": p,
                                "type": "PyTorch",
                                "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                                "compatible": True,
                                "algorithm": exp,
                                "status": "Ready for Evaluation"
                            })

        return checkpoints

    @classmethod
    def list_gameplay_recordings(cls) -> List[Dict[str, Any]]:
        """Scans for available gameplay recordings (GIFs / MP4s)."""
        recordings = []
        video_dirs = [
            os.path.join(PROJECT_ROOT, "outputs", "videos"),
            os.path.join(PROJECT_ROOT, "legacy_baseline", "media")
        ]
        
        seen = set()
        for v_dir in video_dirs:
            if os.path.exists(v_dir):
                for f in sorted(os.listdir(v_dir)):
                    if f.endswith(('.gif', '.mp4')) and f not in seen:
                        seen.add(f)
                        p = os.path.join(v_dir, f)
                        # Extract episode number if available
                        ep = None
                        if "Ep" in f:
                            try:
                                ep = int(f.split("Ep")[1].split(".")[0])
                            except Exception:
                                pass
                        recordings.append({
                            "filename": f,
                            "path": p,
                            "episode": ep,
                            "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)
                        })
        return sorted(recordings, key=lambda x: (x["episode"] if x["episode"] is not None else 999999))

    @classmethod
    def save_config_file(cls, algorithm: str, updates: Dict[str, Any], target_path: Optional[str] = None) -> str:
        """
        Saves updated hyperparameters into a real YAML configuration file.
        """
        if target_path is None:
            target_path = os.path.join(PROJECT_ROOT, "config", f"{algorithm.lower()}.yaml")
            
        base_path = os.path.join(PROJECT_ROOT, "config", "default.yaml")
        config = Config.load(base_path)
        config.algorithm = algorithm.lower()
        
        # Merge updates
        config.merge_from_dict(updates)
        config.save_yaml(target_path)
        return target_path


class LiveGameSession:
    """
    Manages an active Pong game session using the actual Gymnasium Atari environment.
    Runs real agent inference on each frame.
    """
    def __init__(self, agent_type: str = "trained_reinforce", checkpoint_path: Optional[str] = None):
        self.agent_type = agent_type
        self.env = None
        self.agent = None
        self.obs = None
        self.step_count = 0
        self.player_score = 0
        self.opponent_score = 0
        self.cumulative_reward = 0.0
        self.last_action = 0
        self.last_up_prob = 0.5
        self.is_done = False
        self.last_frame = None
        
        self._init_session(checkpoint_path)

    def _init_session(self, checkpoint_path: Optional[str]):
        # Create un-preprocessed renderable environment for crisp visuals
        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
        self.obs, _ = self.env.reset(seed=42)
        self.last_frame = self.env.render()

        if self.agent_type == "trained_reinforce":
            if checkpoint_path is None:
                # Prefer PyTorch .pth, fallback to save.p
                pth_path = os.path.join(PROJECT_ROOT, "checkpoints", "reinforce_karpathy_6000ep.pth")
                legacy_p = os.path.join(PROJECT_ROOT, "legacy_baseline", "checkpoints", "save.p")
                checkpoint_path = pth_path if os.path.exists(pth_path) else legacy_p
                
            self.agent = KarpathyPolicyAgent(checkpoint_path=checkpoint_path)
        elif self.agent_type == "random":
            self.agent = None
        else:
            self.agent = None

    def reset(self, seed: Optional[int] = 42):
        self.obs, _ = self.env.reset(seed=seed)
        if self.agent and hasattr(self.agent, 'reset_state'):
            self.agent.reset_state()
        self.step_count = 0
        self.player_score = 0
        self.opponent_score = 0
        self.cumulative_reward = 0.0
        self.last_action = 0
        self.last_up_prob = 0.5
        self.is_done = False
        self.last_frame = self.env.render()
        return self.last_frame

    def step(self) -> Dict[str, Any]:
        """
        Executes one environment step using the actual agent policy.
        Returns the step info including the rendered RGB frame.
        """
        if self.is_done:
            self.reset()

        # Step 4 frames with the chosen action to match standard Atari 4-frame skip
        if self.agent is not None:
            action_result = self.agent.select_action(self.obs, evaluate=True)
            action = int(action_result)
            self.last_up_prob = getattr(self.agent, "last_up_prob", 0.5)
        else:
            # Random agent
            action = np.random.choice([0, 2, 3])
            self.last_up_prob = 0.5

        self.last_action = action
        step_reward = 0.0
        
        # In Atari Pong, repeat action for 4 frames
        for _ in range(4):
            obs, r, term, trunc, info = self.env.step(action)
            step_reward += r
            if term or trunc:
                self.is_done = True
                break
                
        self.obs = obs
        self.cumulative_reward += step_reward
        self.step_count += 1

        if step_reward > 0:
            self.player_score += 1
        elif step_reward < 0:
            self.opponent_score += 1

        self.last_frame = self.env.render()
        
        return {
            "frame": self.last_frame,
            "action": action,
            "action_name": BackendService.ACTION_NAMES.get(action, f"Action {action}"),
            "up_prob": self.last_up_prob,
            "reward": step_reward,
            "cumulative_reward": self.cumulative_reward,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "step_count": self.step_count,
            "is_done": self.is_done
        }

    def close(self):
        if self.env:
            self.env.close()

