"""
Reinforcement Learning agents for PongRL Pro.

Implements three RL algorithms:
- REINFORCE (Monte Carlo Policy Gradient)
- DQN (Deep Q-Network with Double DQN)
- PPO (Proximal Policy Optimization)
- KarpathyPolicyAgent (Trained REINFORCE agent from Karpathy's Pong baseline)

All agents inherit from BaseAgent and share a common interface.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from abc import ABC, abstractmethod
import os
import pickle

from src.model import PolicyNetwork, QNetwork, ActorCriticNetwork, KarpathyPolicyNetwork
from src.replay import ReplayBuffer, RolloutBuffer, EpisodeMemory


class ActionResult(int):
    """
    Action integer return value that also supports tuple unpacking:
    For DQN/REINFORCE: (action, log_prob)
    For PPO: (action, log_prob, value)
    """
    def __new__(cls, action: int, log_prob: float = 0.0, value: float = 0.0, n_items: int = 2):
        obj = super().__new__(cls, int(action))
        obj.action = int(action)
        obj.log_prob = log_prob
        obj.value = value
        obj._n_items = n_items
        return obj

    def __iter__(self):
        if self._n_items == 3:
            return iter((self.action, self.log_prob, self.value))
        return iter((self.action, self.log_prob))

    def __getitem__(self, idx):
        items = (self.action, self.log_prob, self.value) if self._n_items == 3 else (self.action, self.log_prob)
        return items[idx]


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    Provides a common interface for action selection, learning updates,
    model persistence, and hyperparameter reporting.
    """
    
    def __init__(self, state_shape=None, num_actions=None, config=None):
        # Support flexible constructor: Agent(config) or Agent(state_shape, num_actions, config)
        if hasattr(state_shape, "algorithm") and config is None:
            config = state_shape
            self.config = config
            self.state_shape = getattr(config, "obs_shape", (4, 84, 84))
            self.num_actions = getattr(config, "n_actions", 6)
        else:
            self.state_shape = state_shape if state_shape is not None else (4, 84, 84)
            self.num_actions = num_actions if num_actions is not None else 6
            self.config = config

        self.device = torch.device(
            self.config.device if (self.config and self.config.device not in ["auto", None])
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.learning_rate = self.config.training.get('learning_rate', 1e-4) if (self.config and hasattr(self.config, 'training')) else 1e-4

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """Select an action given the current state."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Perform a learning update. Returns dict of loss metrics."""
        pass

    def save(self, path: str) -> None:
        """Save model weights and optimizer state to a checkpoint file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'algorithm': getattr(self.config, 'algorithm', 'unknown') if self.config else 'unknown',
        }
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint file (PyTorch or pickle)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['network_state_dict'])
                if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif isinstance(checkpoint, dict) and 'W1' in checkpoint and 'W2' in checkpoint:
                # Karpathy pickle weights loaded via torch
                if hasattr(self.network, 'load_numpy_weights'):
                    self.network.load_numpy_weights(checkpoint['W1'], checkpoint['W2'])
            else:
                self.network.load_state_dict(checkpoint)
        except Exception:
            # Fallback for Python pickle weights
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'W1' in data and 'W2' in data:
                if hasattr(self.network, 'load_numpy_weights'):
                    self.network.load_numpy_weights(data['W1'], data['W2'])
                else:
                    raise ValueError(f"Model architecture is not compatible with Karpathy weights: {type(self.network)}")
            else:
                raise

    def get_hyperparams(self) -> dict:
        """Return a dictionary of the agent's hyperparameters."""
        base = {
            "algorithm": getattr(self.config, 'algorithm', 'unknown') if self.config else 'unknown',
            "learning_rate": self.learning_rate,
        }
        if self.config and hasattr(self.config, 'training'):
            base.update(self.config.training)
        return base


class ReinforceAgent(BaseAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    
    Uses full episode returns to compute policy gradients with entropy regularization.
    """
    
    def __init__(self, state_shape=None, num_actions=None, config=None):
        super().__init__(state_shape, num_actions, config)
        in_c = self.state_shape[0] if isinstance(self.state_shape, (tuple, list)) else self.state_shape
        self.network = PolicyNetwork(in_c, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.memory = EpisodeMemory(str(self.device))
        self.gamma = self.config.training.get('gamma', 0.99) if self.config else 0.99
        self.entropy_coef = self.config.training.get('entropy_coef', 0.01) if self.config else 0.01

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.network(state_tensor)
        dist = Categorical(probs)
        
        if evaluate:
            action = torch.argmax(probs, dim=-1).item()
            return ActionResult(action, 0.0, n_items=2)
            
        action = dist.sample()
        return ActionResult(action.item(), dist.log_prob(action).item(), n_items=2)

    def update(self) -> dict:
        if not self.memory.rewards:
            return {}

        returns = self.memory.compute_returns(self.gamma)
        log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.device)
        
        policy_loss = -(log_probs * returns).mean()
        
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        full_probs = self.network(states)
        full_dist = Categorical(full_probs)
        entropy = full_dist.entropy().mean()
        
        loss = policy_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        max_grad = self.config.training.get('max_grad_norm', 0.5) if self.config else 0.5
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad)
        self.optimizer.step()
        
        self.memory.reset()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
        }


class DQNAgent(BaseAgent):
    """
    Double Deep Q-Network (Double DQN) agent.
    """
    
    def __init__(self, state_shape=None, num_actions=None, config=None):
        super().__init__(state_shape, num_actions, config)
        in_c = self.state_shape[0] if isinstance(self.state_shape, (tuple, list)) else self.state_shape
        
        self.network = QNetwork(in_c, self.num_actions).to(self.device)
        self.target_network = QNetwork(in_c, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        buffer_cap = self.config.training.get('buffer_size', 100_000) if self.config else 100_000
        self.memory = ReplayBuffer(buffer_cap, self.state_shape, str(self.device))
        
        self.gamma = self.config.training.get('gamma', 0.99) if self.config else 0.99
        self.batch_size = self.config.training.get('batch_size', 32) if self.config else 32
        self.epsilon = self.config.training.get('exploration_initial_eps', 1.0) if self.config else 1.0
        self.epsilon_min = self.config.training.get('exploration_final_eps', 0.01) if self.config else 0.01
        total_steps = self.config.training.get('total_timesteps', 1_000_000) if self.config else 1_000_000
        explore_fraction = self.config.training.get('exploration_fraction', 0.1) if self.config else 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / max(1, total_steps * explore_fraction)
        self.target_update_freq = self.config.training.get('target_update_freq', 1000) if self.config else 1000
        self.steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        if not evaluate and np.random.random() < self.epsilon:
            return ActionResult(np.random.randint(self.num_actions), 0.0, n_items=2)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values, dim=-1).item()
            return ActionResult(action, 0.0, n_items=2)

    def update(self) -> dict:
        if len(self.memory) < self.batch_size:
            return {}
            
        batch = self.memory.sample(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        with torch.no_grad():
            next_actions = self.network(next_states).argmax(dim=-1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        max_grad = self.config.training.get('max_grad_norm', 0.5) if self.config else 0.5
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
        return {'loss': loss.item(), 'epsilon': self.epsilon}


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(self, state_shape=None, num_actions=None, config=None):
        super().__init__(state_shape, num_actions, config)
        in_c = self.state_shape[0] if isinstance(self.state_shape, (tuple, list)) else self.state_shape
        self.network = ActorCriticNetwork(in_c, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        n_steps = self.config.training.get('n_steps', 128) if self.config else 128
        self.memory = RolloutBuffer(n_steps, self.state_shape, str(self.device))
        
        self.gamma = self.config.training.get('gamma', 0.99) if self.config else 0.99
        self.gae_lambda = self.config.training.get('gae_lambda', 0.95) if self.config else 0.95
        self.clip_range = self.config.training.get('clip_range', 0.2) if self.config else 0.2
        self.ent_coef = self.config.training.get('ent_coef', 0.01) if self.config else 0.01
        self.vf_coef = self.config.training.get('vf_coef', 0.5) if self.config else 0.5
        self.n_epochs = self.config.training.get('n_epochs', 4) if self.config else 4
        self.batch_size = self.config.training.get('batch_size', 64) if self.config else 64

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            if evaluate:
                action = torch.argmax(probs, dim=-1).item()
                return ActionResult(action, 0.0, value.item(), n_items=3)
                
            action = dist.sample()
            return ActionResult(action.item(), dist.log_prob(action).item(), value.item(), n_items=3)

    def update(self, next_value: float = 0.0) -> dict:
        returns, advantages = self.memory.compute_gae(
            next_value, self.gamma, self.gae_lambda
        )
        
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, idx in self.memory.get_batches(self.batch_size):
                logits, values = self.network(states)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * advantages[idx]
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns[idx])
                
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                max_grad = self.config.training.get('max_grad_norm', 0.5) if self.config else 0.5
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad)
                self.optimizer.step()
                
                total_loss_sum += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                n_updates += 1
                
        self.memory.reset()
        
        n = max(1, n_updates)
        return {
            'loss': total_loss_sum / n,
            'policy_loss': policy_loss_sum / n,
            'value_loss': value_loss_sum / n,
        }


class KarpathyPolicyAgent(BaseAgent):
    """
    Trained REINFORCE agent using Karpathy's Pong baseline architecture.
    Operates on 80x80 frame differences and controls Pong paddle via action 2 (UP) and 3 (DOWN).
    """
    def __init__(self, checkpoint_path=None, config=None):
        super().__init__(state_shape=(6400,), num_actions=6, config=config)
        self.network = KarpathyPolicyNetwork(6400, 200).to(self.device)
        self.prev_frame = None
        self.last_up_prob = 0.5
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Preprocess 210x160x3 raw Atari frame to 80x80 binary vector (6400 elements).
        """
        if frame.ndim == 3 and frame.shape[0] == 210:
            frame = frame[35:195]  # crop to 160x160x3
            frame = frame[::2, ::2, 0]  # downsample to 80x80
            frame = frame.astype(np.float32)
            frame[frame == 144] = 0.0
            frame[frame == 109] = 0.0
            frame[frame != 0] = 1.0
            return frame.ravel()
        elif frame.ndim == 3 and frame.shape[0] == 4:
            # Stacked 84x84 observation: take the latest frame and crop/resize to 80x80
            latest = frame[-1]
            if latest.shape == (84, 84):
                latest = latest[2:82, 2:82]  # center 80x80
            return (latest > 0).astype(np.float32).ravel()
        elif frame.ndim == 1 and frame.shape[0] == 6400:
            return frame.astype(np.float32)
        else:
            flat = frame.flatten()
            if len(flat) >= 6400:
                return flat[:6400].astype(np.float32)
            padded = np.zeros(6400, dtype=np.float32)
            padded[:len(flat)] = flat
            return padded

    def reset_state(self):
        """Reset internal frame differencing state."""
        self.prev_frame = None
        self.last_up_prob = 0.5

    def select_action(self, state: np.ndarray, evaluate: bool = True):
        cur_frame = self.preprocess_frame(state)
        diff = cur_frame - self.prev_frame if self.prev_frame is not None else np.zeros(6400, dtype=np.float32)
        self.prev_frame = cur_frame

        diff_t = torch.from_numpy(diff).float().to(self.device)
        with torch.no_grad():
            up_prob = self.network(diff_t).item()
            
        self.last_up_prob = up_prob
        
        # In Pong, 2 is UP, 3 is DOWN
        if evaluate:
            action = 2 if up_prob > 0.5 else 3
        else:
            action = 2 if np.random.uniform() < up_prob else 3
            
        return ActionResult(action, float(up_prob), n_items=2)

    def update(self, *args, **kwargs) -> dict:
        return {}
