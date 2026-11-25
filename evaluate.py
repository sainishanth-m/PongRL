"""
CLI entry point for evaluating trained PongRL Pro agents.

Loads a trained model checkpoint and runs evaluation episodes,
optionally recording gameplay as MP4 video and/or GIF.

Usage:
    python evaluate.py --checkpoint checkpoints/reinforce_karpathy_6000ep.pth --episodes 5
    python evaluate.py --checkpoint best_model.pth --episodes 5 --record --gif eval.gif
"""
import argparse
import os
import torch
from src.config import Config
from src.evaluator import Evaluator
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PongRL Pro agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --checkpoint checkpoints/reinforce_karpathy_6000ep.pth --episodes 5
  python evaluate.py --checkpoint best_model.pth --record --video-folder eval_videos
  python evaluate.py --checkpoint best_model.pth --gif gameplay.gif --episodes 5
        """,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to base config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--algo-config", type=str,
        help="Path to algorithm-specific config YAML override",
    )
    parser.add_argument(
        "--algorithm", type=str, choices=["dqn", "reinforce", "ppo"],
        help="Algorithm override (dqn, reinforce, ppo)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint file (.pth or .p)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record gameplay as MP4 video using Gymnasium RecordVideo wrapper",
    )
    parser.add_argument(
        "--video-folder", type=str, default="outputs/videos",
        help="Directory to save recorded MP4 videos (default: outputs/videos)",
    )
    parser.add_argument(
        "--gif", type=str,
        help="Path to save GIF recording (e.g., gameplay.gif)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Auto-detect algorithm if checkpoint has metadata
    kwargs = {'device': get_device(args.device)}
    algo_override = args.algorithm
    if not algo_override and os.path.exists(args.checkpoint):
        if "reinforce" in args.checkpoint.lower() or "karpathy" in args.checkpoint.lower() or args.checkpoint.endswith(".p"):
            algo_override = "reinforce"
        elif "ppo" in args.checkpoint.lower():
            algo_override = "ppo"
        elif "dqn" in args.checkpoint.lower():
            algo_override = "dqn"
            
    if algo_override:
        kwargs['algorithm'] = algo_override
        if not args.algo_config and os.path.exists(f"config/{algo_override}.yaml"):
            args.algo_config = f"config/{algo_override}.yaml"
            
    # Load configuration
    config = Config.load(args.config, args.algo_config, **kwargs)
    set_seed(config.seed)
    
    # Create evaluator with checkpoint
    evaluator = Evaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        record_video=args.record,
        video_folder=args.video_folder,
    )
    
    # Run evaluation
    record_gif = bool(args.gif)
    gif_path = args.gif if args.gif else "gameplay.gif"
    
    print(f"{'='*60}")
    print(f"  PongRL Pro — Agent Evaluation")
    print(f"{'='*60}")
    print(f"  Algorithm:  {config.algorithm}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Record MP4: {args.record}")
    print(f"  Record GIF: {record_gif}")
    print(f"  Device:     {config.device}")
    print(f"{'='*60}\n")
    
    results = evaluator.evaluate(
        episodes=args.episodes,
        record_gif=record_gif,
        gif_path=gif_path,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean Reward:   {results['mean_reward']:>8.2f} ± {results['std_reward']:.2f}")
    print(f"  Min Reward:    {results.get('min_reward', 'N/A'):>8}")
    print(f"  Max Reward:    {results.get('max_reward', 'N/A'):>8}")
    print(f"  Mean Length:   {results['mean_length']:>8.1f}")
    
    if 'rewards' in results:
        print(f"\n  Per-Episode Rewards:")
        for i, r in enumerate(results['rewards']):
            print(f"    Episode {i+1:>3d}: {r:>8.1f}")
    
    print(f"{'='*60}")
    
    if args.record:
        print(f"\n  Videos saved to: {args.video_folder}/")
    if record_gif:
        print(f"  GIF saved to:    {gif_path}")


if __name__ == "__main__":
    main()
