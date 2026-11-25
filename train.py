import argparse
import os
from src.config import Config
from src.trainer import Trainer
from src.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train PongRL Pro Agent")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to base config")
    parser.add_argument("--algo-config", type=str, help="Path to algorithm specific config")
    parser.add_argument("--algorithm", type=str, choices=["dqn", "reinforce", "ppo"], help="Algorithm to use")
    parser.add_argument("--total-timesteps", type=int, help="Total timesteps to train")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment-name", type=str, default="pong_experiment", help="Name of experiment")
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.algorithm: kwargs['algorithm'] = args.algorithm
    if args.total_timesteps: kwargs['training.total_timesteps'] = args.total_timesteps
    if args.seed: kwargs['seed'] = args.seed
    
    device = get_device(args.device)
    kwargs['device'] = device
    
    # Auto-select algo config if not provided but algo is
    algo_conf = args.algo_config
    if not algo_conf and args.algorithm:
        algo_conf = f"config/{args.algorithm}.yaml"
        
    config = Config.load(args.config, algo_conf, **kwargs)
    
    set_seed(config.seed)
    
    trainer = Trainer(config, args.experiment_name)
    
    if args.resume and os.path.exists(args.resume):
        trainer.agent.load(args.resume)
        
    trainer.train()

if __name__ == "__main__":
    main()
