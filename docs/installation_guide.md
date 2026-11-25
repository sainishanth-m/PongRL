# Installation Guide

This guide provides step-by-step instructions to set up the PongRL Pro framework on your local machine.

## Prerequisites
- Python 3.10 or higher
- Git
- (Optional but recommended) NVIDIA GPU with CUDA Toolkit installed for hardware acceleration.

## Step 1: Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/username/PongRL-Pro.git
cd PongRL-Pro
```

## Step 2: Virtual Environment Setup
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

This command will install core dependencies including `torch`, `gymnasium`, `numpy`, and `streamlit`.

## Step 4: Atari ROM Setup
To play Atari games like Pong, Gymnasium requires the original ROM files. The installation of `gymnasium[accept-rom-license]` usually handles this automatically. If you encounter errors regarding missing ROMs, run:

```bash
AutoROM --accept-license
```

## CPU-Only Notes
If you do not have a dedicated GPU, PyTorch will automatically default to CPU execution. Note that training Deep Reinforcement Learning algorithms on raw image data using only a CPU will be significantly slower. You can explicitly force CPU usage in the config file by setting `device: "cpu"`.

## Verification Steps
To verify that the installation was successful, run a short evaluation script using a dummy or untrained model:

```bash
python evaluate.py --config config/default.yaml --episodes 1
```

**Expected Output:**
You should see logging output indicating the environment was created successfully, followed by the reward obtained in one episode.

## Troubleshooting FAQ

**Q1: `ModuleNotFoundError: No module named 'gymnasium'`**
A: Ensure your virtual environment is activated and you have run `pip install -r requirements.txt`.

**Q2: PyTorch is not utilizing my GPU.**
A: Check your PyTorch installation. You may need to install a specific CUDA version of PyTorch from the official website (https://pytorch.org/get-started/locally/).

**Q3: Missing ROM errors.**
A: Ensure you have accepted the ROM license by running `AutoROM --accept-license`.

**Q4: Execution fails with memory errors.**
A: Deep learning requires significant RAM/VRAM. Try reducing the `batch_size` or `replay_buffer_size` in the YAML configuration files.

**Q5: Streamlit dashboard doesn't load.**
A: Ensure port 8501 is not blocked by your firewall. Run `streamlit run dashboard/app.py` and check the console for specific binding errors.
