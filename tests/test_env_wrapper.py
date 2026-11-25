import pytest
from src.env_wrapper import make_pong_env

def test_env_creation(mock_config):
    env = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    assert env is not None

def test_env_obs_shape(mock_config):
    env = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    obs, _ = env.reset()
    assert obs.shape == mock_config.obs_shape

def test_env_action_space(mock_config):
    env = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    assert env.action_space.n == mock_config.n_actions

def test_env_reset_step_tuples(mock_config):
    env = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    obs, info = env.reset()
    assert isinstance(info, dict)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)

def test_env_seeding(mock_config):
    env1 = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    env2 = make_pong_env(mock_config, record_video=False, video_folder="", seed=42)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert (obs1 == obs2).all()

def test_env_record_video(mock_config, tmp_path):
    env = make_pong_env(mock_config, record_video=True, video_folder=str(tmp_path), seed=42)
    assert env is not None
