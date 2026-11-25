# Expected Results and Analysis Framework

## Experimental Setup Summary
The empirical evaluation compares REINFORCE, DQN, and PPO on the `PongNoFrameskip-v4` environment. All algorithms utilize the exact same CNN architecture for feature extraction and the same preprocessing pipeline to ensure fairness. Hardware used for baseline generation: NVIDIA RTX GPU, 16GB RAM.

## Expected Results from Literature
Based on established DRL literature:
- **REINFORCE**: We expect very slow learning, potentially failing to reach a positive score within 5M frames due to high gradient variance, even with a baseline.
- **DQN**: We expect a slow start due to the necessity of filling the replay buffer, followed by a steady increase in score, typically converging to ~20 around 2-3M frames.
- **PPO**: We anticipate rapid initial learning, outperforming DQN in the first 1M frames, owing to the stability of the clipped objective and the efficiency of multiple epochs over the collected rollout data.

## Analysis Framework
Results will be analyzed across the following dimensions:
1. **Sample Efficiency**: How many environment interactions are required to reach a score of +15?
2. **Wall-clock Time**: How long does training take in real time? PPO is often faster per frame than DQN due to batched forward passes during rollout collection.
3. **Stability**: Variance in the evaluation score across different random seeds.

## Comparison Table Template

| Metric | REINFORCE | DQN | PPO |
|--------|-----------|-----|-----|
| Time to Score > 0 | TBD | TBD | TBD |
| Time to Score > 15 | TBD | TBD | TBD |
| Final Mean Score | TBD | TBD | TBD |
| Training Time (hrs) | TBD | TBD | TBD |

## Discussion Points for Final Report
- Why did algorithm X outperform algorithm Y in this specific environment?
- How did hyperparameter choices (e.g., PPO clip range, DQN target update frequency) impact the observed stability?
- Does visual complexity in Atari necessitate algorithmic changes compared to state-vector environments (like MuJoCo)?
