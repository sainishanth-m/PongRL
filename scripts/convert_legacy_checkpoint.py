"""
Converts legacy Karpathy Pong policy gradient weights (save.p) to PyTorch .pth checkpoint.
"""
import os
import sys
import pickle
import torch
import numpy as np

# Ensure root of project is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import KarpathyPolicyNetwork

def convert_checkpoint(
    source_p: str = None,
    dest_pth: str = None
):
    if source_p is None:
        source_p = os.path.join(PROJECT_ROOT, "legacy_baseline", "checkpoints", "save.p")
    if dest_pth is None:
        dest_pth = os.path.join(PROJECT_ROOT, "checkpoints", "reinforce_karpathy_6000ep.pth")

    if not os.path.exists(source_p):
        raise FileNotFoundError(f"Source pickle not found: {source_p}")

    print(f"Loading legacy weights from: {source_p}")
    with open(source_p, "rb") as f:
        data = pickle.load(f)

    w1 = data['W1']
    w2 = data['W2']
    print(f"Loaded W1 shape: {w1.shape}, W2 shape: {w2.shape}")

    model = KarpathyPolicyNetwork(6400, 200)
    model.load_numpy_weights(w1, w2)

    os.makedirs(os.path.dirname(dest_pth), exist_ok=True)
    checkpoint = {
        'network_state_dict': model.state_dict(),
        'algorithm': 'reinforce',
        'architecture': 'karpathy_mlp_200',
        'episodes_trained': 6000,
        'best_running_reward': -1.47,
        'description': 'Trained 6000-episode REINFORCE policy network on Pong'
    }
    torch.save(checkpoint, dest_pth)
    print(f"Successfully exported PyTorch checkpoint to: {dest_pth}")

if __name__ == "__main__":
    convert_checkpoint()
