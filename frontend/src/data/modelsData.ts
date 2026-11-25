export interface ModelBenchmark {
  algorithm: string;
  category: string;
  architecture: string;
  parameters: number;
  asymptoticReward: number;
  sampleEfficiencyEpisodes: number;
  inferenceLatencyMs: number;
  memoryUsageMb: number;
  stabilityScore: number;
  status: string;
  pros: string[];
  cons: string[];
}

export const MODEL_BENCHMARKS: ModelBenchmark[] = [
  {
    algorithm: "REINFORCE",
    category: "Policy Gradient (Monte Carlo)",
    architecture: "Karpathy MLP (6,400 → 200 → 1)",
    parameters: 1280201,
    asymptoticReward: -1.47,
    sampleEfficiencyEpisodes: 3200,
    inferenceLatencyMs: 1.42,
    memoryUsageMb: 48,
    stabilityScore: 0.78,
    status: "Fully Converged (6,001 Episodes)",
    pros: [
      "Lowest forward pass inference latency (1.42 ms)",
      "Minimal host memory footprint (48 MB RAM)",
      "Direct policy optimization without Q-value overestimation"
    ],
    cons: [
      "Higher variance in raw Monte Carlo returns",
      "Requires standardized advantage normalization"
    ]
  },
  {
    algorithm: "DQN",
    category: "Value-Based (Q-Learning)",
    architecture: "Nature CNN (3 Conv Layers + FC 512)",
    parameters: 1687206,
    asymptoticReward: -2.10,
    sampleEfficiencyEpisodes: 2600,
    inferenceLatencyMs: 3.85,
    memoryUsageMb: 1410,
    stabilityScore: 0.86,
    status: "Benchmarked Reference",
    pros: [
      "Sample-efficient off-policy transitions",
      "Experience replay buffer breaks sequential frame correlation",
      "Quasi-static target network stabilizes TD-targets"
    ],
    cons: [
      "Higher memory consumption for 100k+ transition buffer",
      "Susceptible to overestimating action utilities"
    ]
  },
  {
    algorithm: "PPO",
    category: "Actor-Critic (Trust Region)",
    architecture: "Nature CNN Dual Head (Actor + Critic)",
    parameters: 1690285,
    asymptoticReward: 1.85,
    sampleEfficiencyEpisodes: 1450,
    inferenceLatencyMs: 4.12,
    memoryUsageMb: 180,
    stabilityScore: 0.95,
    status: "Benchmarked Reference",
    pros: [
      "Highest policy convergence stability",
      "Clipped surrogate objective prevents destructive policy collapse",
      "Generalized Advantage Estimation (GAE) balances bias and variance"
    ],
    cons: [
      "Dual forward passes for policy and value heads",
      "Slightly higher compute cost per environment step"
    ]
  }
];

export const CHECKPOINTS_INVENTORY = [
  {
    name: "reinforce_karpathy_6000ep.pth",
    format: "PyTorch State Dict",
    architecture: "KarpathyPolicyNetwork",
    episodes: 6001,
    peakReward: -1.47,
    sizeMb: 5.12,
    compatible: true,
    description: "Converged 6,000-episode weights converted directly from legacy baseline."
  },
  {
    name: "legacy/save.p",
    format: "NumPy Pickle",
    architecture: "Karpathy 200 Hidden Neurons",
    episodes: 6001,
    peakReward: -1.47,
    sizeMb: 4.88,
    compatible: true,
    description: "Original NumPy baseline checkpoint loaded via KarpathyPolicyAgent adapter."
  },
  {
    name: "dqn_nature_cnn_atari.pth",
    format: "PyTorch State Dict",
    architecture: "NatureCNN (4, 84, 84)",
    episodes: 2600,
    peakReward: -2.10,
    sizeMb: 6.75,
    compatible: true,
    description: "Standard DeepMind Nature CNN architecture benchmark checkpoint."
  }
];
