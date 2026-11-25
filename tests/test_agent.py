import pytest
import torch
from src.agent import ReinforceAgent, DQNAgent, PPOAgent, KarpathyPolicyAgent

def test_reinforce_creation(mock_config):
    agent = ReinforceAgent(mock_config)
    assert agent is not None
    assert agent.network is not None

def test_dqn_creation(mock_config):
    agent = DQNAgent(mock_config)
    assert agent is not None
    assert agent.network is not None

def test_ppo_creation(mock_config):
    agent = PPOAgent(mock_config)
    assert agent is not None
    assert agent.network is not None

def test_agent_action_selection(mock_config, sample_observations):
    agent = DQNAgent(mock_config)
    action = agent.select_action(sample_observations)
    assert isinstance(action, int)
    assert 0 <= int(action) < 6

def test_agent_save_load(mock_config, tmp_path):
    agent = DQNAgent(mock_config)
    path = str(tmp_path / "model.pth")
    agent.save(path)
    
    agent2 = DQNAgent(mock_config)
    agent2.load(path)
    for p1, p2 in zip(agent.network.parameters(), agent2.network.parameters()):
        assert torch.allclose(p1, p2)

def test_dqn_epsilon_decay(mock_config):
    agent = DQNAgent(mock_config)
    assert hasattr(agent, 'epsilon')
    assert agent.epsilon <= 1.0

def test_agent_hyperparams(mock_config):
    agent = PPOAgent(mock_config)
    assert agent.config == mock_config
    params = agent.get_hyperparams()
    assert "learning_rate" in params

def test_ppo_update(mock_config):
    agent = PPOAgent(mock_config)
    assert hasattr(agent, 'update')

def test_reinforce_update(mock_config):
    agent = ReinforceAgent(mock_config)
    assert hasattr(agent, 'update')
