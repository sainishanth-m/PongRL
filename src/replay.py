import numpy as np
import torch
from typing import Tuple, Dict, List, Optional, Any


class SampleBatch(dict):
    """
    Dictionary of sampled transitions that can also be unpacked as a 5-tuple:
    (states, actions, rewards, next_states, dones).
    """
    def __iter__(self):
        return iter((
            self['states'],
            self['actions'],
            self['rewards'],
            self['next_states'],
            self['dones']
        ))


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    Stores visual observations compactly as uint8 (0-255) to conserve memory.
    """
    def __init__(self, capacity: int = 50_000, state_shape: tuple = (4, 84, 84), device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape if isinstance(state_shape, tuple) else (state_shape,)
        
        # Use uint8 for Atari pixel frames to prevent 10+ GB allocations
        self.dtype = np.uint8 if (len(self.state_shape) >= 2 and max(self.state_shape) >= 40) else np.float32
        
        self.states = np.zeros((capacity, *self.state_shape), dtype=self.dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *self.state_shape), dtype=self.dtype)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        state_arr = np.asarray(state)
        # Adapt capacity arrays if state shape was inferred
        if state_arr.shape != self.states.shape[1:]:
            self.state_shape = state_arr.shape
            self.dtype = np.uint8 if (len(self.state_shape) >= 2 and max(self.state_shape) >= 40) else np.float32
            self.states = np.zeros((self.capacity, *self.state_shape), dtype=self.dtype)
            self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=self.dtype)

        self.states[self.pos] = state_arr.astype(self.dtype)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = np.asarray(next_state).astype(self.dtype)
        self.dones[self.pos] = float(done)
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> SampleBatch:
        n = min(batch_size, self.size)
        indices = np.random.choice(self.size, n, replace=False if self.size >= n else True)
        
        batch = SampleBatch({
            'states': torch.FloatTensor(self.states[indices].astype(np.float32)).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices].astype(np.float32)).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        })
        return batch

    def clear(self):
        """Reset replay buffer to empty state."""
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """Buffer for on-policy algorithms like PPO."""
    def __init__(
        self,
        capacity: int = 128,
        state_shape: tuple = (4, 84, 84),
        device: str = "cpu",
        obs_shape: Optional[tuple] = None,
        action_dim: int = 1,
        buffer_size: Optional[int] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        **kwargs
    ):
        self.capacity = buffer_size if buffer_size is not None else capacity
        self.buffer_size = self.capacity
        self.state_shape = obs_shape if obs_shape is not None else state_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.step = 0
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.step = 0

    def clear(self):
        self.reset()

    def add(self, state, action, reward, value=0.0, log_prob=0.0, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.step = len(self.states)

    def compute_gae(self, next_value: float, gamma: Optional[float] = None, gae_lambda: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gamma if gamma is None else gamma
        l = self.gae_lambda if gae_lambda is None else gae_lambda
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + g * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + g * l * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return torch.tensor(returns, dtype=torch.float32).to(self.device), \
               torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def get_batches(self, batch_size: int):
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        for start_idx in range(0, n_samples, batch_size):
            idx = indices[start_idx:start_idx + batch_size]
            yield states[idx], actions[idx], old_log_probs[idx], idx

    def __len__(self) -> int:
        return len(self.states)


class EpisodeMemory:
    """Memory for episodic algorithms like REINFORCE."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def clear(self):
        self.reset()

    def add(self, state, action, reward, log_prob=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self, gamma: float = 0.99, normalize: bool = True) -> torch.Tensor:
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        if normalize and len(returns) > 1:
            std = returns.std(unbiased=False)
            if std > 1e-8:
                returns = (returns - returns.mean()) / (std + 1e-8)
        elif normalize and len(returns) == 1:
            returns = returns - returns.mean()
            
        return returns
