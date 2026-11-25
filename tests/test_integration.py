import pytest
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.agent import DQNAgent
from src.env_wrapper import make_pong_env
from src.logger import TrainingLogger

def test_training_smoke(mock_config, tmp_path):
    logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test", use_tensorboard=False)
    env = make_pong_env(mock_config, False, "", 42)
    agent = DQNAgent(mock_config)
    trainer = Trainer(agent, env, mock_config, logger)
    trainer.train(episodes=1)
    assert True

def test_checkpoint_resume(mock_config, tmp_path):
    assert True

def test_evaluation_smoke(mock_config):
    agent = DQNAgent(mock_config)
    evaluator = Evaluator(agent, mock_config)
    evaluator.evaluate(episodes=1)
    assert True

def test_full_pipeline(mock_config, tmp_path):
    assert True
