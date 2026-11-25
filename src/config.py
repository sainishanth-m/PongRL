"""
Configuration loader for PongRL Pro.

Loads YAML configuration files, merges base and algorithm-specific configs,
supports CLI argument overrides, and validates configuration values.
"""
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


# Valid values for validation
VALID_ALGORITHMS = {"dqn", "reinforce", "ppo"}
VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}
VALID_ARCHITECTURES = {"nature_cnn", "simple_mlp"}


@dataclass
class Config:
    """
    Configuration container for PongRL Pro.
    
    Loads hierarchical YAML configs with support for algorithm-specific
    overrides and CLI argument overrides.
    
    Attributes:
        device: Compute device ('cpu', 'cuda', 'mps', or 'auto').
        seed: Random seed for reproducibility.
        algorithm: RL algorithm to use ('dqn', 'reinforce', 'ppo').
        env_id: Gymnasium environment identifier.
        training: Dictionary of training hyperparameters.
        env: Dictionary of environment configuration.
        model: Dictionary of model architecture configuration.
    """
    device: str = "cpu"
    seed: int = 42
    algorithm: str = "dqn"
    env_id: str = "ALE/Pong-v5"
    
    training: Dict[str, Any] = field(default_factory=lambda: {
        "total_timesteps": 1_000_000,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 100_000,
        "checkpoint_interval": 50_000,
        "log_interval": 10,
        "eval_interval": 25_000,
        "eval_episodes": 5,
        "save_freq": 50_000,
        "max_grad_norm": 0.5,
    })
    env: Dict[str, Any] = field(default_factory=lambda: {
        "frame_stack": 4,
        "frame_skip": 4,
        "screen_size": 84,
        "grayscale": True,
        "noop_max": 30,
        "difficulty": 0,
    })
    model: Dict[str, Any] = field(default_factory=lambda: {
        "architecture": "nature_cnn",
        "hidden_size": 512,
    })

    # Convenient property accessors for commonly used hyperparameters
    @property
    def learning_rate(self) -> float:
        return self.training.get("learning_rate", 1e-4)

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.training["learning_rate"] = value

    @property
    def gamma(self) -> float:
        return self.training.get("gamma", 0.99)

    @gamma.setter
    def gamma(self, value: float) -> None:
        self.training["gamma"] = value

    @property
    def batch_size(self) -> int:
        return self.training.get("batch_size", 32)

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.training["batch_size"] = value

    @property
    def total_timesteps(self) -> int:
        return self.training.get("total_timesteps", 1_000_000)

    @total_timesteps.setter
    def total_timesteps(self, value: int) -> None:
        self.training["total_timesteps"] = value

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        stack = self.env.get("frame_stack", self.env.get("stack_size", 4))
        size = self.env.get("screen_size", 84)
        return (stack, size, size)

    @property
    def n_actions(self) -> int:
        return 6  # Standard discrete action space for Atari Pong

    @classmethod
    def load(cls, base_path: str, override_path: str = None, **kwargs) -> 'Config':
        """
        Load configuration from YAML files with optional overrides.
        
        Args:
            base_path: Path to the base YAML configuration file.
            override_path: Optional path to algorithm-specific YAML override.
            **kwargs: CLI argument overrides (supports dotted keys like 'training.learning_rate').
            
        Returns:
            A fully merged and validated Config instance.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base config file not found: {base_path}")
            
        with open(base_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            
        if override_path and os.path.exists(override_path):
            with open(override_path, 'r') as f:
                override_data = yaml.safe_load(f) or {}
                data = cls._merge_dicts(data, override_data)
                
        # Apply CLI keyword overrides
        for k, v in kwargs.items():
            if v is not None:
                if '.' in k:
                    # Support dotted keys like 'training.learning_rate'
                    section, key = k.split('.', 1)
                    if section not in data:
                        data[section] = {}
                    data[section][key] = v
                else:
                    data[k] = v
        
        # Filter to only known dataclass fields to prevent TypeError
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
                    
        config = cls(**filtered_data)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load Config directly from a single YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML config not found: {path}")
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}
            
        # If top level contains flat keys like learning_rate, map them to sections
        base = cls()
        base_dict = base.to_dict()
        for k, v in raw.items():
            if k in base_dict:
                if isinstance(v, dict) and isinstance(base_dict[k], dict):
                    base_dict[k].update(v)
                else:
                    base_dict[k] = v
            elif k in base_dict["training"]:
                base_dict["training"][k] = v
            elif k in base_dict["env"]:
                base_dict["env"][k] = v
            elif k in base_dict["model"]:
                base_dict["model"][k] = v
                
        config = cls.from_dict(base_dict)
        return config

    def save_yaml(self, path: str) -> None:
        """Save configuration dictionary to a YAML file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def merge_from_dict(self, override: Dict[str, Any]) -> None:
        """Merge a dictionary of parameters into this Config instance."""
        for k, v in override.items():
            if hasattr(self, k) and not isinstance(getattr(self, k), dict):
                setattr(self, k, v)
            elif k in self.training:
                self.training[k] = v
            elif k in self.env:
                self.env[k] = v
            elif k in self.model:
                self.model[k] = v
            elif k == "training" and isinstance(v, dict):
                self.training.update(v)
            elif k == "env" and isinstance(v, dict):
                self.env.update(v)
            elif k == "model" and isinstance(v, dict):
                self.model.update(v)

    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{self.algorithm}'. "
                f"Must be one of: {VALID_ALGORITHMS}"
            )
        
        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Must be one of: {VALID_DEVICES}"
            )
            
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(
                f"Invalid seed '{self.seed}'. Must be a non-negative integer."
            )
            
        arch = self.model.get("architecture", "nature_cnn")
        if arch not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Invalid architecture '{arch}'. "
                f"Must be one of: {VALID_ARCHITECTURES}"
            )
            
        lr = self.training.get("learning_rate", 1e-4)
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(
                f"Invalid learning_rate '{lr}'. Must be a positive number."
            )
            
        gamma = self.training.get("gamma", 0.99)
        if not isinstance(gamma, (int, float)) or not (0 <= gamma <= 1):
            raise ValueError(
                f"Invalid gamma '{gamma}'. Must be between 0 and 1."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "device": self.device,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "env_id": self.env_id,
            "training": dict(self.training),
            "env": dict(self.env),
            "model": dict(self.model),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create a Config from a dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    @staticmethod
    def _merge_dicts(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge override dict into base dict.
        
        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).
            
        Returns:
            Merged dictionary.
        """
        merged = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = Config._merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged
