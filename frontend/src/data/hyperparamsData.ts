export interface HyperparamGroup {
  name: string;
  description: string;
  params: { key: string; label: string; value: string | number | boolean; type: string; desc: string }[];
}

export const HYPERPARAMETERS_DATA: HyperparamGroup[] = [
  {
    name: "General Environment Configuration",
    description: "Parameters governing the Gymnasium Arcade Learning Environment setup",
    params: [
      { key: "env_id", label: "Environment Identifier", value: "ALE/Pong-v5", type: "string", desc: "Target Atari game environment in Farama Gymnasium" },
      { key: "frame_stack", label: "Frame Stack Depth", value: 4, type: "number", desc: "Number of consecutive frames stacked for Nature CNN" },
      { key: "frame_skip", label: "Action Repeat (Frame Skip)", value: 4, type: "number", desc: "Number of game frames an agent action is held" },
      { key: "screen_size", label: "Downscaled Screen Resolution", value: "84 × 84", type: "string", desc: "Processed spatial observation dimensions" },
      { key: "reward_clip", label: "Reward Clipping", value: true, type: "boolean", desc: "Clips rewards to {-1, 0, +1} to standardize gradients" },
    ]
  },
  {
    name: "REINFORCE (Karpathy Policy Gradient)",
    description: "Parameters utilized for the converged 6,001-episode training run",
    params: [
      { key: "learning_rate", label: "Learning Rate (α)", value: "1e-3 (0.001)", type: "string", desc: "RMSprop initial step size for policy updates" },
      { key: "gamma", label: "Discount Factor (γ)", value: 0.99, type: "number", desc: "Weight assigned to future rewards across a rally" },
      { key: "hidden_dim", label: "Hidden Layer Neurons", value: 200, type: "number", desc: "Capacity of the fully connected ReLU layer" },
      { key: "batch_size", label: "Batch Episode Size", value: 10, type: "number", desc: "Episodes aggregated before performing weight backpropagation" },
      { key: "decay_rate", label: "RMSprop Decay Rate", value: 0.99, type: "number", desc: "Moving average parameter for gradient history" },
      { key: "input_dim", label: "State Input Vector", value: 6400, type: "number", desc: "Flattened 80×80 binarized difference frame" },
    ]
  },
  {
    name: "Deep Q-Networks (DQN)",
    description: "Parameters governing off-policy Q-learning with Nature CNN",
    params: [
      { key: "buffer_size", label: "Replay Buffer Capacity", value: "100,000", type: "string", desc: "Maximum transitions retained in uint8 memory" },
      { key: "batch_size", label: "Mini-batch Sampling Size", value: 64, type: "number", desc: "Number of sampled transitions per SGD step" },
      { key: "learning_starts", label: "Exploration Warmup Steps", value: 1000, type: "number", desc: "Random exploration steps prior to gradient descent" },
      { key: "target_update_freq", label: "Target Network Sync Interval", value: 1000, type: "number", desc: "Timesteps between target Q-network weight copies" },
      { key: "exploration_initial", label: "Initial Epsilon (ε)", value: 1.0, type: "number", desc: "Starting probability of exploratory random actions" },
      { key: "exploration_final", label: "Final Epsilon (ε)", value: 0.02, type: "number", desc: "Asymptotic exploratory floor" },
    ]
  },
  {
    name: "Proximal Policy Optimization (PPO)",
    description: "Parameters controlling trust-region actor-critic policy updates",
    params: [
      { key: "n_steps", label: "Rollout Horizon Steps", value: 2048, type: "number", desc: "Transitions collected per policy optimization cycle" },
      { key: "clip_range", label: "Surrogate Clipping Ratio (ε)", value: 0.2, type: "number", desc: "PPO trust region clipping bound [1-ε, 1+ε]" },
      { key: "n_epochs", label: "PPO Epochs per Rollout", value: 10, type: "number", desc: "SGD passes over the collected rollout buffer" },
      { key: "gae_lambda", label: "GAE Lambda (λ)", value: 0.95, type: "number", desc: "Generalized Advantage Estimator bias-variance balance" },
      { key: "vf_coef", label: "Value Function Loss Weight", value: 0.5, type: "number", desc: "Coefficient weighting the critic MSE loss" },
      { key: "ent_coef", label: "Entropy Regularization Weight", value: 0.01, type: "number", desc: "Exploration bonus discouraging premature certainty" },
    ]
  }
];
