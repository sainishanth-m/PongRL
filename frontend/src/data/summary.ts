export interface ProjectSummary {
  name: string;
  tagline: string;
  subtitle: string;
  description: string;
  recordedEpisodes: number;
  peakRunningReward: number;
  initialBaselineReward: number;
  finalReward: number;
  activeModels: number;
  trainingStatus: string;
  environment: string;
  observationSpace: string;
  actionSpace: number;
  primaryCheckpoint: string;
  algorithms: string[];
  techStack: { name: string; category: string; description: string }[];
}

export const PROJECT_SUMMARY: ProjectSummary = {
  name: "PongRL Pro",
  tagline: "Intelligent Game Agent Using Deep Reinforcement Learning",
  subtitle: "An Intelligent Pong Agent Powered by Deep Reinforcement Learning and Computer Vision.",
  description:
    "PongRL Pro implements and benchmarks deep reinforcement learning agents in the Gymnasium Atari Pong environment. The agent learns directly from raw visual observations using a convolutional neural network and optimizes its policy using reinforcement learning algorithms.",
  recordedEpisodes: 6001,
  peakRunningReward: -1.47,
  initialBaselineReward: -20.0,
  finalReward: -1.59,
  activeModels: 3,
  trainingStatus: "Completed (6,001 Episodes)",
  environment: "ALE/Pong-v5 (Gymnasium)",
  observationSpace: "210×160 RGB (Preprocessed to 80×80 Difference)",
  actionSpace: 6,
  primaryCheckpoint: "reinforce_karpathy_6000ep.pth",
  algorithms: ["REINFORCE", "DQN", "PPO"],
  techStack: [
    { name: "Python 3.10+", category: "Core Backend", description: "Primary runtime for model inference and training workflows" },
    { name: "PyTorch 2.x", category: "Deep Learning", description: "GPU/CPU accelerated tensor ops and autograd neural architectures" },
    { name: "Gymnasium", category: "Simulation", description: "Farama Foundation Atari 2600 environment interface" },
    { name: "ALE Atari Pong", category: "Engine", description: "Cycle-accurate Stella 6502 machine simulation at 60 FPS" },
    { name: "Next.js & React", category: "Frontend", description: "Modern React framework deployed serverlessly on Vercel" },
    { name: "Tailwind CSS", category: "Styling", description: "Utility-first design system with minimal, clean typography" },
    { name: "Recharts", category: "Visualization", description: "Declarative SVG charts for empirical learning curves" },
    { name: "HTML5 Canvas", category: "Game Engine", description: "Zero-latency 60 FPS in-browser Pong court" },
  ],
};
