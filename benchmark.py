import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config
from src.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple algorithms")
    parser.add_argument("--checkpoints", nargs='+', required=True, help="List of checkpoints: path1 path2 ...")
    parser.add_argument("--algorithms", nargs='+', required=True, help="List of algorithms matching checkpoints: dqn ppo ...")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per model")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.algorithms):
        raise ValueError("Number of checkpoints must match number of algorithms")
        
    results = []
    
    for ckpt, algo in zip(args.checkpoints, args.algorithms):
        print(f"Evaluating {algo} from {ckpt}...")
        
        config = Config.load("config/default.yaml", f"config/{algo}.yaml", algorithm=algo)
        evaluator = Evaluator(config, ckpt)
        
        metrics = evaluator.evaluate(episodes=args.episodes)
        
        results.append({
            'Algorithm': algo.upper(),
            'Mean Reward': metrics['mean_reward'],
            'Std Reward': metrics['std_reward'],
            'Mean Length': metrics['mean_length']
        })
        
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    print(df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Mean Reward', data=df, capsize=.1)
    plt.title('Benchmark Results: Mean Reward by Algorithm')
    plt.savefig('benchmark_plot.png')
    print("Saved plot to benchmark_plot.png")

if __name__ == "__main__":
    main()
