import torch
import numpy as np
import random
import os
import imageio
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(requested_device: str = "cpu") -> str:
    """Return correct device prioritizing requested, falling back if needed."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        return requested_device
    return "cpu"

def compute_discounted_returns(rewards, gamma):
    """Compute discounted returns for a list of rewards."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def frames_to_gif(frames, path, fps=30):
    """Save a sequence of frames as a GIF."""
    imageio.mimsave(path, frames, fps=fps)

def plot_training_curves(metrics_csv, out_path):
    """Plot training curves from CSV data."""
    pass # Implementation in evaluator/logger

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
