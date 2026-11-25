"""
Environment wrapper for PongRL Pro.

Creates and configures Gymnasium Atari environments with standard
preprocessing wrappers for Deep Reinforcement Learning.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import ale_py
import numpy as np

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


class SeededResetWrapper(gym.Wrapper):
    """Ensures consistent initial seed on reset if not explicitly provided."""
    def __init__(self, env, default_seed: int = 42):
        super().__init__(env)
        self.default_seed = default_seed

    def reset(self, *, seed=None, options=None):
        effective_seed = seed if seed is not None else self.default_seed
        return self.env.reset(seed=effective_seed, options=options)


def make_pong_env(
    env_id = "ALE/Pong-v5",
    seed: int = 42,
    render_mode: str = None,
    record_video: bool = False,
    video_folder: str = "videos",
    difficulty: int = 0,
) -> gym.Env:
    """
    Create and wrap the Pong environment with standard Atari preprocessing.
    
    The preprocessing pipeline applies:
    1. No-op reset (up to 30 random no-op actions at start)
    2. Frame skipping (repeat action for 4 frames, max-pool last 2)
    3. Grayscale conversion
    4. Resize to 84x84
    5. Stack last 4 frames along channel dimension
    
    Args:
        env_id: Gymnasium environment identifier, or a Config object.
        seed: Random seed for reproducibility.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        record_video: If True, wraps environment with RecordVideo.
        video_folder: Directory to save recorded videos.
        difficulty: ALE difficulty level (0=easy, 1=medium, 2=hard).
        
    Returns:
        Wrapped Gymnasium environment ready for RL training.
    """
    if hasattr(env_id, "env_id"):
        config = env_id
        actual_env_id = getattr(config, "env_id", "ALE/Pong-v5")
        actual_seed = getattr(config, "seed", seed)
        env_dict = getattr(config, "env", {})
        difficulty = env_dict.get("difficulty", difficulty)
    else:
        actual_env_id = str(env_id) if env_id else "ALE/Pong-v5"
        actual_seed = seed

    # Ensure empty string is converted to None
    effective_render_mode = render_mode if (render_mode and render_mode != "") else None
    if record_video and effective_render_mode is None:
        effective_render_mode = "rgb_array"
    
    env = gym.make(
        actual_env_id,
        render_mode=effective_render_mode,
        frameskip=1,
        difficulty=difficulty,
    )
    
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    
    env = FrameStackObservation(env, stack_size=4)
    
    if actual_seed is not None:
        env = SeededResetWrapper(env, default_seed=actual_seed)
        env.action_space.seed(actual_seed)
        env.observation_space.seed(actual_seed)

    if record_video:
        env = RecordVideo(
            env,
            video_folder=video_folder if video_folder else "outputs/videos",
            episode_trigger=lambda ep_id: True,
            name_prefix="pongrl-agent",
        )
        
    return env
