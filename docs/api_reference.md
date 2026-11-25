# API Reference

This document provides a comprehensive reference for all modules, classes, and functions within the `src` directory of the PongRL Pro framework.

---

## Module: `src.config`

### Class `Config`
Data class responsible for holding the configuration parameters for the RL framework.
- **Attributes**:
  - `device` (str): Hardware device to use ('cpu', 'cuda', 'mps', 'auto').
  - `seed` (int): Random seed.
  - `algorithm` (str): Name of the algorithm ('ppo', 'dqn', 'reinforce').
  - `env_id` (str): Gymnasium environment ID.
  - `training` (dict): Dictionary of training hyperparameters.
  - `env` (dict): Dictionary of environment configurations.
  - `model` (dict): Dictionary of model architecture configurations.

- **Methods**:
  - `load(path: str) -> Config`: Loads configuration from a YAML file.
  - `_merge_dicts(d1: dict, d2: dict) -> dict`: Recursively merges two dictionaries.

---

## Module: `src.model`

### Class `NatureCNN`
Convolutional neural network architecture based on the original DQN paper.
- **Constructor**: `__init__(self, in_channels: int)`
- **Methods**: `forward(self, x: torch.Tensor) -> torch.Tensor`

### Class `SimpleMLP`
A simple Multi-Layer Perceptron.
- **Constructor**: `__init__(self, input_size: int, hidden_size: int)`
- **Methods**: `forward(self, x: torch.Tensor) -> torch.Tensor`

### Class `PolicyNetwork`
Outputs a probability distribution over actions.
- **Constructor**: `__init__(self, in_channels: int, num_actions: int)`
- **Methods**: `forward(self, x: torch.Tensor) -> torch.Tensor`

### Class `QNetwork`
Outputs Q-values for all possible actions given a state.
- **Constructor**: `__init__(self, in_channels: int, num_actions: int)`
- **Methods**: `forward(self, x: torch.Tensor) -> torch.Tensor`

### Class `ActorCriticNetwork`
Shared representation network with two heads: one for policy (actor) and one for value (critic).
- **Constructor**: `__init__(self, in_channels: int, num_actions: int)`
- **Methods**: `forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

---

## Module: `src.agent`

### Class `BaseAgent`
Abstract base class for all RL agents.
- **Constructor**: `__init__(self, state_shape: tuple, num_actions: int, config: Config)`
- **Methods**:
  - `select_action(self, state: np.ndarray, evaluate: bool = False) -> int`
  - `update(self) -> dict`
  - `save(self, path: str) -> None`
  - `load(self, path: str) -> None`
  - `get_hyperparams(self) -> dict`

### Class `ReinforceAgent` (Inherits `BaseAgent`)
Implementation of the REINFORCE algorithm.
- **Constructor**: `__init__(self, state_shape: tuple, num_actions: int, config: Config)`

### Class `DQNAgent` (Inherits `BaseAgent`)
Implementation of Deep Q-Network.
- **Constructor**: `__init__(self, state_shape: tuple, num_actions: int, config: Config)`

### Class `PPOAgent` (Inherits `BaseAgent`)
Implementation of Proximal Policy Optimization.
- **Constructor**: `__init__(self, state_shape: tuple, num_actions: int, config: Config)`

---

## Module: `src.replay`

### Class `ReplayBuffer`
Standard circular experience replay buffer used for off-policy algorithms like DQN.
- **Constructor**: `__init__(self, capacity: int, state_shape: tuple, device: torch.device)`
- **Methods**:
  - `push(self, state, action, reward, next_state, done)`
  - `sample(self, batch_size: int)`
  - `__len__(self)`

### Class `RolloutBuffer`
Buffer for on-policy algorithms like PPO, storing fixed-length trajectories.
- **Constructor**: `__init__(self, capacity: int, state_shape: tuple, device: torch.device)`
- **Methods**:
  - `push(self, state, action, reward, next_state, done, log_prob, value)`
  - `compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float)`
  - `get_generator(self, batch_size: int)`
  - `clear(self)`

### Class `EpisodeMemory`
Stores transitions for a single episode, typically used by REINFORCE.
- **Constructor**: `__init__(self, device: torch.device)`
- **Methods**:
  - `push(self, log_prob, reward)`
  - `clear(self)`

---

## Module: `src.env_wrapper`

### Function `make_pong_env`
Creates and wraps the Pong environment with standard Atari preprocessing.
- **Signature**: `make_pong_env(env_id: str, seed: int, render_mode: str = None, record_video: bool = False, video_folder: str = None) -> gym.Env`
- **Description**: Applies NoopReset, MaxAndSkip, EpisodicLife, FireReset, WarpFrame, ClipReward, and FrameStack wrappers.

---

## Module: `src.trainer`

### Class `Trainer`
Orchestrates the training loop for the RL agent.
- **Constructor**: `__init__(self, config: Config, experiment_name: str)`
- **Methods**:
  - `train(self) -> None`: Main training loop. Manages environment interaction, agent updates, and logging.

---

## Module: `src.evaluator`

### Class `Evaluator`
Handles the evaluation of trained agents.
- **Constructor**: `__init__(self, config: Config, checkpoint_path: str)`
- **Methods**:
  - `evaluate(self, episodes: int = 10, record_gif: bool = False, gif_path: str = None) -> float`: Evaluates the agent and optionally records gameplay.

---

## Module: `src.logger`

### Class `TrainingLogger`
Handles writing metrics to TensorBoard, CSV, and console.
- **Constructor**: `__init__(self, log_dir: str, experiment_name: str)`
- **Methods**:
  - `log_step(self, step: int, metrics: dict) -> None`
  - `info(self, message: str) -> None`
  - `plot_metrics(self, save_path: str) -> None`
  - `close(self) -> None`

---

## Module: `src.utils`

- `set_seed(seed: int) -> None`: Fixes random seeds for NumPy, Python random, and PyTorch.
- `get_device(device_str: str) -> torch.device`: Resolves the compute device.
- `compute_discounted_returns(rewards: list, gamma: float) -> list`: Calculates discounted returns for Monte Carlo methods.
- `frames_to_gif(frames: list, path: str, fps: int = 30) -> None`: Converts a list of numpy arrays to a GIF file.
- `format_time(seconds: float) -> str`: Formats a duration in seconds to a human-readable string (e.g., HH:MM:SS).
