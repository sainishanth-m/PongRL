import pytest
import yaml
import os
from src.config import Config

def test_config_load_yaml(tmp_path):
    yaml_content = {"learning_rate": 0.05, "batch_size": 128}
    filepath = tmp_path / "config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(yaml_content, f)
    config = Config.from_yaml(str(filepath))
    assert config.learning_rate == 0.05

def test_config_merge():
    config = Config()
    config.merge_from_dict({"gamma": 0.95})
    assert config.gamma == 0.95

def test_config_validation():
    config = Config()
    config.validate()

def test_config_validation_invalid():
    config = Config()
    config.learning_rate = -1
    with pytest.raises(ValueError):
        config.validate()

def test_config_roundtrip(tmp_path):
    config = Config()
    config.learning_rate = 0.02
    d = config.to_dict()
    filepath = tmp_path / "test.yaml"
    with open(filepath, "w") as f:
        yaml.dump(d, f)
    config2 = Config.from_yaml(str(filepath))
    assert config2.learning_rate == 0.02

def test_config_device_default():
    config = Config()
    assert config.device in ["cpu", "cuda", "mps"]

def test_config_to_dict():
    config = Config()
    d = config.to_dict()
    assert isinstance(d, dict)
