export const PROJECT_INFO_DATA = {
  problemStatement:
    "Autonomous decision making from high-dimensional visual observations without handcrafted heuristics is a foundational milestone in Artificial Intelligence. The Arcade Learning Environment (ALE) Atari 2600 game Pong presents core challenges: delayed sparse rewards (+1 upon scoring, -1 upon conceding), extreme input dimensionality (100,800 scalar values per RGB frame), and the temporal credit assignment problem across hundreds of intermediary paddle actions.",
  objectives: [
    "Design and modularize three foundational Deep Reinforcement Learning algorithms (REINFORCE, Deep Q-Networks, and Proximal Policy Optimization) natively in PyTorch.",
    "Formulate an efficient 5-stage computer vision preprocessing pipeline reducing observation dimensionality by 95.9% while encoding directional velocity vectors via temporal frame differencing.",
    "Mitigate catastrophic host Out-Of-Memory exceptions on commodity workstations through an optimized uint8 circular replay buffer architecture, reducing memory consumption from 22.5 GB to 1.41 GB (93.75% reduction).",
    "Empirically train and validate policy convergence over 6,001 consecutive episodes, tracking variance reduction and asymptotic progression from -20.4 to -1.47 reward.",
    "Deploy a zero-latency interactive web research suite enabling human evaluators to play directly against the trained PyTorch agent in real time."
  ],
  methodology: [
    { title: "Markov Decision Process", desc: "Formulated under S, A, P, R, gamma tuple where state st is the 80×80 difference tensor Δst = st - st-1." },
    { title: "Policy Gradient Theorem", desc: "Analytical gradient calculation ∇θJ(θ) = E[Σ ∇θ log π(at|st) Gt] with standardized Monte Carlo advantage normalization." },
    { title: "Value Approximation", desc: "Deep Q-Networks minimizing temporal difference error with quasi-static target networks and experience replay." },
    { title: "Trust Region Optimization", desc: "PPO clipped surrogate objective preventing destructive policy collapse under large learning steps." }
  ],
  references: [
    { id: 1, text: "V. Mnih, K. Kavukcuoglu, D. Silver, et al., 'Human-level control through deep reinforcement learning,' Nature, vol. 518, no. 7540, pp. 529–533, 2015." },
    { id: 2, text: "R. J. Williams, 'Simple statistical gradient-following algorithms for connectionist reinforcement learning,' Machine Learning, vol. 8, no. 3–4, pp. 229–256, 1992." },
    { id: 3, text: "J. Schulman, F. Wolski, P. Dhariwal, et al., 'Proximal policy optimization algorithms,' arXiv preprint arXiv:1707.06347, 2017." },
    { id: 4, text: "M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling, 'The Arcade Learning Environment: An evaluation platform for general agents,' JAIR, vol. 47, pp. 253–279, 2013." },
    { id: 5, text: "A. Karpathy, 'Deep Reinforcement Learning: Pong from Pixels,' Andrej Karpathy Blog, May 2016." },
    { id: 6, text: "R. S. Sutton and A. G. Barto, 'Reinforcement Learning: An Introduction,' 2nd ed., MIT Press, Cambridge, MA, 2018." },
    { id: 7, text: "M. Towers, et al., 'Gymnasium: A standard interface for reinforcement learning environments,' arXiv:2407.17032, 2023." },
    { id: 8, text: "A. Paszke, et al., 'PyTorch: An imperative style, high-performance deep learning library,' NeurIPS, 2019." }
  ],
  vivaQuestions: [
    {
      q: "How does the agent infer ball velocity without frame stacking?",
      a: "We compute the temporal frame difference: Δst = st - st-1. In this binary matrix, +1 represents the current ball position and -1 represents the ball position in the immediately preceding frame. The directional offset between +1 and -1 directly supplies the network with the velocity vector (dx/dt, dy/dt) with zero multi-frame memory overhead."
    },
    {
      q: "Why was the replay buffer converted from float32 to uint8?",
      a: "Raw Atari frames are 8-bit unsigned integers [0, 255]. Storing frames naively as 32-bit floats consumes 4 bytes per pixel, meaning a 1-million frame buffer requires 22.5 GB of RAM, which crashes 16 GB workstations with an ArrayMemoryError. Storing them in uint8 requires only 1 byte per pixel, dropping memory consumption to 1.41 GB—a 93.75% memory reduction."
    },
    {
      q: "Why does the episode reward plateau near -1.47 instead of +21.0 in REINFORCE?",
      a: "A score of -1.47 indicates the agent is winning approximately 48.8% of all individual points against the built-in Atari heuristic CPU (which plays with mathematical reflex). Because Pong matches end at 21, a score of -1.47 translates to an average match score of 19–21 or 20–21, with sustained rallies of 18+ returns."
    },
    {
      q: "What is the core difference between Policy Gradient (REINFORCE) and Q-Learning (DQN)?",
      a: "DQN is value-based; it approximates the expected state-action utility Q(s, a) and chooses actions greedily via argmax. REINFORCE is policy-based; it directly parameterizes the action probability distribution π(a|s) and optimizes it via gradient ascent on expected cumulative returns using the Policy Gradient Theorem."
    }
  ]
};
