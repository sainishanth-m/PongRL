"""
Shared pytest fixtures for PongRL Pro test suite.
"""
import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import MagicMock
import gymnasium as gym


@pytest.fixture
def mock_config():
    """Create a Config object with minimal settings for fast tests."""
    from src.config import Config
    config = Config(
        device="cpu",
        seed=42,
        algorithm="dqn",
        env_id="ALE/Pong-v5",
        training={
            "total_timesteps": 1000,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32,
            "buffer_size": 1000,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.1,
            "target_update_freq": 100,
            "learning_starts": 100,
            "train_freq": 4,
            "save_freq": 500,
            "max_grad_norm": 0.5,
            # PPO-specific
            "n_steps": 128,
            "n_epochs": 4,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gae_lambda": 0.95,
            # REINFORCE-specific
            "entropy_coef": 0.01,
        },
        env={
            "frame_stack": 4,
            "frame_skip": 4,
            "screen_size": 84,
            "grayscale": True,
            "noop_max": 30,
            "difficulty": 0,
        },
        model={
            "architecture": "nature_cnn",
            "hidden_size": 512,
        },
    )
    return config


@pytest.fixture
def mock_config_reinforce(mock_config):
    """Config variant for REINFORCE agent."""
    mock_config.algorithm = "reinforce"
    return mock_config


@pytest.fixture
def mock_config_ppo(mock_config):
    """Config variant for PPO agent."""
    mock_config.algorithm = "ppo"
    return mock_config


@pytest.fixture
def tmp_dirs():
    """Provide a temporary directory for save/load tests."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def sample_cnn_observation():
    """Random tensor shaped like a stacked Atari frame (4, 84, 84)."""
    return np.random.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)


@pytest.fixture
def sample_observations(sample_cnn_observation):
    """Alias for sample_cnn_observation."""
    return sample_cnn_observation


@pytest.fixture
def sample_cnn_batch():
    """Batch of CNN observations (B, 4, 84, 84)."""
    return torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.float32)


@pytest.fixture
def mock_env():
    """A simple Gymnasium environment for testing without Atari ROMs."""
    env = gym.make("CartPole-v1")
    yield env
    env.close()
