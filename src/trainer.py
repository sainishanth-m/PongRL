import time
import os
from tqdm import tqdm
import numpy as np
from typing import Optional

from src.env_wrapper import make_pong_env
from src.agent import BaseAgent, DQNAgent, ReinforceAgent, PPOAgent
from src.logger import TrainingLogger
from src.utils import format_time


class Trainer:
    """
    Training pipeline orchestrator for PongRL Pro agents.
    
    Supports REINFORCE, DQN, and PPO with automatic logging,
    checkpoint saving, and device management.
    """
    def __init__(self, config=None, experiment_name: str = "experiment", agent=None, env=None, logger=None):
        if isinstance(config, BaseAgent):
            # Called as Trainer(agent, env, config, logger)
            self.agent = config
            self.env = experiment_name
            self.config = agent
            self.logger = logger if logger else (env if hasattr(env, 'log_step') else TrainingLogger("runs", "experiment"))
            self.experiment_name = getattr(self.config, 'experiment_name', 'experiment')
        else:
            self.config = config
            self.experiment_name = experiment_name
            self.env = make_pong_env(config.env_id, config.seed)
            
            state_shape = self.env.observation_space.shape
            num_actions = self.env.action_space.n
            
            if config.algorithm == "dqn":
                self.agent = DQNAgent(state_shape, num_actions, config)
            elif config.algorithm == "reinforce":
                self.agent = ReinforceAgent(state_shape, num_actions, config)
            elif config.algorithm == "ppo":
                self.agent = PPOAgent(state_shape, num_actions, config)
            else:
                raise ValueError(f"Unknown algorithm: {config.algorithm}")
                
            self.logger = logger if logger else TrainingLogger("runs", experiment_name)
            
        self.total_timesteps = self.config.training.get('total_timesteps', 1000000) if self.config else 1000000
        self.save_dir = os.path.join("runs", self.experiment_name, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self, episodes: Optional[int] = None):
        algo = getattr(self.config, 'algorithm', 'unknown') if self.config else 'unknown'
        env_id = getattr(self.config, 'env_id', 'ALE/Pong-v5') if self.config else 'ALE/Pong-v5'
        device = getattr(self.config, 'device', 'cpu') if self.config else 'cpu'
        
        self.logger.info(f"Starting training {algo} on {env_id}")
        self.logger.info(f"Device: {device}")
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episodes_completed = 0
        
        start_time = time.time()
        best_reward = float('-inf')
        
        max_steps = self.total_timesteps if episodes is None else min(self.total_timesteps, 5000)
        
        try:
            pbar = tqdm(total=max_steps, desc="Training")
            step = 0
            
            while step < max_steps:
                if algo == "ppo":
                    action, log_prob, value = self.agent.select_action(np.array(obs))
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    self.agent.memory.add(np.array(obs), action, reward, value, log_prob, done)
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    
                    n_steps_trigger = self.config.training.get('n_steps', 128) if self.config else 128
                    if len(self.agent.memory.states) >= n_steps_trigger:
                        _, _, next_value = self.agent.select_action(np.array(next_obs), evaluate=True)
                        update_info = self.agent.update(next_value)
                        self.logger.log_step(step, update_info)
                
                elif algo == "dqn":
                    action, _ = self.agent.select_action(np.array(obs))
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    self.agent.memory.add(np.array(obs), action, reward, np.array(next_obs), done)
                    
                    learn_starts = self.config.training.get('learning_starts', 10000) if self.config else 10000
                    train_freq = self.config.training.get('train_freq', 4) if self.config else 4
                    if step >= learn_starts and step % train_freq == 0:
                        update_info = self.agent.update()
                        if update_info and step % 1000 == 0:
                            self.logger.log_step(step, update_info)
                            
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    
                else:  # reinforce or others
                    action, log_prob = self.agent.select_action(np.array(obs))
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    self.agent.memory.add(np.array(obs), action, reward, log_prob)
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        update_info = self.agent.update()
                        if update_info:
                            self.logger.log_step(step, update_info)

                if done:
                    episodes_completed += 1
                    self.logger.log_step(step, {
                        'episode_reward': episode_reward,
                        'episode_length': episode_length
                    })
                    
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        self.agent.save(os.path.join(self.save_dir, "best_model.pth"))
                        
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    
                    if episodes is not None and episodes_completed >= episodes:
                        break
                
                step += 1
                pbar.update(1)
                
                save_f = self.config.training.get('save_freq', 50000) if self.config else 50000
                if step % save_f == 0:
                    self.agent.save(os.path.join(self.save_dir, f"model_step_{step}.pth"))

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user. Saving current model...")
        finally:
            self.agent.save(os.path.join(self.save_dir, "final_model.pth"))
            pbar.close()
            self.env.close()
            self.logger.plot_metrics()
            self.logger.close()
            
            elapsed = time.time() - start_time
            self.logger.info(f"Training completed in {format_time(elapsed)}")
