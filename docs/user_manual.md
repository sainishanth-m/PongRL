# User Manual

Welcome to the PongRL Pro User Manual. This document provides practical instructions on how to use the CLI tools for training, evaluating, and benchmarking RL algorithms.

## 1. Training an Agent

The primary entry point for training is `train.py`.

### Basic Usage
```bash
python train.py --config config/default.yaml --experiment-name my_first_run
```

### CLI Options Explained
- `--config` (Required): Path to the base YAML configuration file.
- `--override` (Optional): Path to a secondary YAML file. Values here will override the base config. Useful for switching algorithms quickly.
- `--algorithm` (Optional): Override the algorithm directly (e.g., `ppo`, `dqn`).
- `--seed` (Optional): Set a specific random seed for reproducibility.
- `--device` (Optional): Force a specific compute device (`cpu`, `cuda`).
- `--experiment-name` (Required): Defines the folder name inside `runs/` where logs, checkpoints, and videos will be saved.

### Algorithm Selection Guide
- **PPO**: Recommended for general use. Fast, stable, and sample-efficient.
- **DQN**: Good for understanding value-based methods. Takes longer to converge but can achieve high final performance.
- **REINFORCE**: Primarily for educational purposes. High variance and difficult to converge on visual tasks.

## 2. Configuration Customization

Configurations are managed via YAML files in the `config/` directory.

Example of changing hyperparameters:
```yaml
# config/my_custom_ppo.yaml
algorithm: ppo
training:
  learning_rate: 0.0003
  batch_size: 128
  total_timesteps: 2000000
```
Run using: `python train.py --config config/default.yaml --override config/my_custom_ppo.yaml ...`

## 3. Evaluation and Video Recording

Use `evaluate.py` to test trained models.

### Basic Evaluation
```bash
python evaluate.py --config config/default.yaml --checkpoint runs/my_first_run/checkpoints/best_model.pth --episodes 20
```

### Recording Gameplay
Add the `--record-gif` flag to save the evaluation as a visual file:
```bash
python evaluate.py --config config/default.yaml --checkpoint runs/my_first_run/checkpoints/best_model.pth --episodes 1 --record-gif
```
The GIF will be saved in the `videos/` directory of your run.

## 4. Dashboard Usage

PongRL Pro includes an interactive Streamlit dashboard for analyzing training logs.

```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` in your browser. You can select different experiment runs from the sidebar to view learning curves, episode lengths, and loss metrics.

## 5. Benchmarking

To compare multiple algorithms automatically, use `benchmark.py`.

```bash
python benchmark.py --algorithms dqn ppo reinforce --timesteps 500000 --base-config config/default.yaml
```
This script will sequentially train all specified algorithms, generate comparative plots, and output a summary table in the terminal.

## 6. Generating Reports

While there is no automated report generator, the data saved in the `runs/` directory (CSV logs, TensorBoard events, saved plots) is structured specifically to make it easy to import into LaTeX documents or Markdown reports. Use the exported PNG files from the dashboard for your presentations.
