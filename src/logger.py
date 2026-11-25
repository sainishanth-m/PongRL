import logging
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TrainingLogger:
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup standard logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(self.log_dir, 'training.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(fh)
        
        # Tensorboard
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        else:
            self.writer = None
        
        self.metrics = []

    def log_step(self, step: int, metrics_dict: dict):
        """Log metrics to console, file, and tensorboard."""
        record = dict(metrics_dict)
        record['step'] = step
        self.metrics.append(record)
        
        # Tensorboard
        if self.writer:
            for key, value in record.items():
                if key != 'step' and isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                
        # File
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def info(self, msg: str):
        self.logger.info(msg)
        print(msg)
        
    def plot_metrics(self):
        """Auto-generate plots from recorded metrics."""
        if not self.metrics:
            return
            
        df = pd.DataFrame(self.metrics)
        
        if 'episode_reward' in df.columns:
            clean_df = df.dropna(subset=['episode_reward'])
            if not clean_df.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(clean_df['step'], clean_df['episode_reward'])
                plt.title('Training Reward over time')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.savefig(os.path.join(self.log_dir, 'reward_plot.png'))
                plt.close()
            
    def close(self):
        if self.writer:
            self.writer.close()
