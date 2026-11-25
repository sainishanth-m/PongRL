import pytest
import torch
from src.model import NatureCNN, SimpleMLP, PolicyNetwork, QNetwork, ActorCriticNetwork

def test_nature_cnn_shapes():
    model = NatureCNN(in_channels=4, n_actions=2, hidden_size=256)
    x = torch.randn(1, 4, 84, 84)
    out = model(x)
    assert out.shape == (1, 2)

def test_simple_mlp_shapes():
    model = SimpleMLP(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(1, 4)
    out = model(x)
    assert out.shape == (1, 2)

def test_policy_network_probabilities():
    model = PolicyNetwork(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(1, 4)
    out = model(x)
    assert torch.allclose(out.sum(dim=-1), torch.tensor(1.0))

def test_q_network_shapes():
    model = QNetwork(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(1, 4)
    out = model(x)
    assert out.shape == (1, 2)

def test_actor_critic_network_shapes():
    model = ActorCriticNetwork(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(1, 4)
    actor_out, critic_out = model(x)
    assert actor_out.shape == (1, 2)
    assert critic_out.shape == (1, 1)

def test_model_save_load(tmp_path):
    model = SimpleMLP(in_channels=4, n_actions=2, hidden_size=64)
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    model2 = SimpleMLP(in_channels=4, n_actions=2, hidden_size=64)
    model2.load_state_dict(torch.load(path))
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

def test_model_gradients():
    model = SimpleMLP(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(1, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None

def test_model_batch_sizes():
    model = SimpleMLP(in_channels=4, n_actions=2, hidden_size=64)
    x = torch.randn(32, 4)
    out = model(x)
    assert out.shape == (32, 2)
