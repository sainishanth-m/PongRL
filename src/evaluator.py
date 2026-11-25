"""
Evaluation module for PongRL Pro.

Provides systematic evaluation of trained RL agents with support for
MP4 video recording, GIF generation, and comprehensive statistics.
"""
import os
import numpy as np
from tqdm import tqdm
from typing import Optional

from src.env_wrapper import make_pong_env
from src.agent import BaseAgent, DQNAgent, ReinforceAgent, PPOAgent, KarpathyPolicyAgent
from src.utils import frames_to_gif


class Evaluator:
    """
    Evaluates trained RL agents on the Pong environment.
    
    Supports MP4 video recording via Gymnasium's RecordVideo wrapper,
    GIF generation from captured frames, and detailed per-episode statistics.
    """
    
    def __init__(
        self,
        config,
        checkpoint_path: Optional[str] = None,
        record_video: bool = False,
        video_folder: str = "outputs/videos",
        agent: Optional[BaseAgent] = None,
    ):
        if isinstance(config, BaseAgent):
            # Called as Evaluator(agent, config)
            self.agent = config
            self.config = checkpoint_path
            self.record_video = record_video
            self.video_folder = video_folder
        else:
            self.config = config
            self.record_video = record_video
            self.video_folder = video_folder
            
            state_shape = (4, 84, 84)
            num_actions = 6
            if agent is not None:
                self.agent = agent
            elif checkpoint_path and (checkpoint_path.endswith(".p") or "karpathy" in checkpoint_path.lower()):
                self.agent = KarpathyPolicyAgent(checkpoint_path=checkpoint_path, config=config)
            else:
                agent_classes = {
                    "dqn": DQNAgent,
                    "reinforce": ReinforceAgent,
                    "ppo": PPOAgent,
                }
                algo = getattr(config, 'algorithm', 'dqn')
                agent_cls = agent_classes.get(algo, DQNAgent)
                self.agent = agent_cls(state_shape, num_actions, config)
                
            if checkpoint_path and os.path.exists(checkpoint_path) and not isinstance(self.agent, KarpathyPolicyAgent):
                self.agent.load(checkpoint_path)

        # Create evaluation environment
        env_id = getattr(self.config, 'env_id', "ALE/Pong-v5") if self.config else "ALE/Pong-v5"
        seed = getattr(self.config, 'seed', 42) if self.config else 42
        self.env = make_pong_env(
            env_id=env_id,
            seed=seed,
            render_mode="rgb_array",
            record_video=record_video,
            video_folder=video_folder,
        )

    def evaluate(
        self,
        episodes: int = 10,
        record_gif: bool = False,
        gif_path: str = "gameplay.gif",
    ) -> dict:
        rewards = []
        lengths = []
        gif_frames = []
        
        for ep in tqdm(range(episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            if hasattr(self.agent, 'reset_state'):
                self.agent.reset_state()
                
            ep_reward = 0
            ep_length = 0
            done = False
            
            if record_gif and ep == 0:
                frame = self.env.render()
                if frame is not None:
                    gif_frames.append(frame)
                
            while not done:
                action_result = self.agent.select_action(
                    np.array(obs), evaluate=True
                )
                action = int(action_result) if isinstance(action_result, (int, float)) else action_result[0]
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_length += 1
                
                if record_gif and ep == 0:
                    frame = self.env.render()
                    if frame is not None:
                        gif_frames.append(frame)
                    
            rewards.append(ep_reward)
            lengths.append(ep_length)
            
        if record_gif and gif_frames:
            os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
            frames_to_gif(gif_frames, gif_path)
            
        self.env.close()
        
        return {
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
            'std_reward': float(np.std(rewards)) if rewards else 0.0,
            'min_reward': float(np.min(rewards)) if rewards else 0.0,
            'max_reward': float(np.max(rewards)) if rewards else 0.0,
            'mean_length': float(np.mean(lengths)) if lengths else 0.0,
            'rewards': rewards,
            'lengths': lengths,
        }
