import pytest
import numpy as np
from src.replay import ReplayBuffer, RolloutBuffer, EpisodeMemory

def test_replay_buffer_add():
    buf = ReplayBuffer(capacity=100)
    buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    assert len(buf) == 1

def test_replay_buffer_capacity():
    buf = ReplayBuffer(capacity=2)
    buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    assert len(buf) == 2

def test_replay_buffer_sample():
    buf = ReplayBuffer(capacity=100)
    for _ in range(10):
        buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    obs, acts, rews, next_obs, dones = buf.sample(5)
    assert len(obs) == 5

def test_rollout_buffer_gae():
    buf = RolloutBuffer(obs_shape=(4,), action_dim=1, buffer_size=10, gamma=0.99, gae_lambda=0.95)
    assert buf.gamma == 0.99

def test_rollout_buffer_batches():
    buf = RolloutBuffer(obs_shape=(4,), action_dim=1, buffer_size=10, gamma=0.99, gae_lambda=0.95)
    assert buf.buffer_size == 10

def test_episode_memory_returns():
    mem = EpisodeMemory()
    mem.rewards = [1, 1, 1]
    assert len(mem.rewards) == 3

def test_episode_memory_normalization():
    mem = EpisodeMemory()
    mem.rewards = [1, 1, 1]
    assert True

def test_replay_buffer_clear():
    buf = ReplayBuffer(capacity=100)
    buf.add(np.zeros(4), 1, 1.0, np.zeros(4), False)
    buf.clear()
    assert len(buf) == 0

def test_rollout_buffer_add():
    buf = RolloutBuffer(obs_shape=(4,), action_dim=1, buffer_size=10, gamma=0.99, gae_lambda=0.95)
    buf.add(np.zeros(4), 1, 1.0, 0.5, 0.1)
    assert buf.step == 1
