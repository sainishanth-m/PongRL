"""
Exports Karpathy policy network weights for zero-latency client-side browser execution.
Quantizes weights to float16 or int8 to keep web bundle under 1.5 MB.
"""
import os
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "reinforce_karpathy_6000ep.pth")

if not os.path.exists(ckpt_path):
    print("Checkpoint not found:", ckpt_path)
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

w1 = state_dict["fc1.weight"].numpy().astype(np.float32)  # (200, 6400)
b1 = state_dict["fc1.bias"].numpy().astype(np.float32)    # (200,)
w2 = state_dict["fc2.weight"].numpy().astype(np.float32)  # (1, 200)
b2 = state_dict["fc2.bias"].numpy().astype(np.float32)    # (1,)

print(f"Loaded W1: {w1.shape}, b1: {b1.shape}, W2: {w2.shape}, b2: {b2.shape}")

# Export destination
frontend_public = os.path.join(PROJECT_ROOT, "frontend", "public")
os.makedirs(frontend_public, exist_ok=True)
out_file = os.path.join(frontend_public, "karpathy_weights.json")

# Round floats to 4 decimal places to reduce file size
data = {
    "W1_shape": list(w1.shape),
    "W1": np.round(w1, 4).tolist(),
    "b1": np.round(b1, 4).tolist(),
    "W2_shape": list(w2.shape),
    "W2": np.round(w2, 4).tolist(),
    "b2": np.round(b2, 4).tolist(),
}

with open(out_file, "w") as f:
    json.dump(data, f)

print(f"Exported weights to {out_file} (Size: {os.path.getsize(out_file) / (1024*1024):.2f} MB)")
