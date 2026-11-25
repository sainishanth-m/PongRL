export const RECORDINGS_DATA = [
  {
    episode: 1,
    title: "Episode 1: Random Jitter (Untrained Baseline)",
    reward: -20.4,
    status: "Exploration Floor",
    filename: "karpathyModelEp1.gif",
    path: "/media/karpathyModelEp1.gif",
    description: "Pure stochastic action sampling. The agent repeatedly jitters, misses serves, and concedes 21 consecutive points within seconds."
  },
  {
    episode: 1000,
    title: "Episode 1,000: Early Contact Learning",
    reward: -19.2,
    status: "Early Coordination",
    filename: "karpathyModelEp1000.gif",
    path: "/media/karpathyModelEp1000.gif",
    description: "Policy begins reinforcing paddle actions that successfully make contact with the ball, reducing high-frequency jitter."
  },
  {
    episode: 2000,
    title: "Episode 2,000: Defensive Tracking Emergence",
    reward: -16.5,
    status: "Tracking Stabilization",
    filename: "karpathyModelEp2000.gif",
    path: "/media/karpathyModelEp2000.gif",
    description: "Agent begins moving toward incoming ball trajectories and executing defensive returns on straightforward trajectories."
  },
  {
    episode: 3000,
    title: "Episode 3,000: Strategic Inflection Point",
    reward: -11.2,
    status: "Inflection Shift",
    filename: "karpathyModelEp3000.gif",
    path: "/media/karpathyModelEp3000.gif",
    description: "Steepest rate of improvement. The agent sustains 5–8 exchange rallies and wins 20% of individual game points."
  },
  {
    episode: 4000,
    title: "Episode 4,000: Angle Deflection & Speed Handling",
    reward: -7.5,
    status: "Offensive Acquisition",
    filename: "karpathyModelEp4000.gif",
    path: "/media/karpathyModelEp4000.gif",
    description: "Discovers edge returns. Striking the ball with paddle extremities imparts sharp diagonal trajectories that challenge the CPU."
  },
  {
    episode: 5000,
    title: "Episode 5,000: Near-Parity Matchplay",
    reward: -3.8,
    status: "Competitive Stabilization",
    filename: "karpathyModelEp5000.gif",
    path: "/media/karpathyModelEp5000.gif",
    description: "Win rate approaches 40%. The agent routinely counters high-speed serves and sustains rallies exceeding 12 exchanges."
  },
  {
    episode: 6000,
    title: "Episode 6,000: Asymptotic Master Policy",
    reward: -1.47,
    status: "Asymptotic Convergence",
    filename: "karpathyModelEp6000.gif",
    path: "/media/karpathyModelEp6000.gif",
    description: "Converged autonomous master. Sustains 18+ returns per rally, scores aces cleanly, and achieves a 48.8% win rate against the Atari CPU."
  }
];
