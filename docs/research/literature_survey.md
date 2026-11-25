# Literature Survey: Deep Reinforcement Learning for Game Playing

## 1. RL Foundations
Reinforcement Learning models the interaction between an agent and its environment as a Markov Decision Process (MDP), characterized by a tuple $(S, A, P, R, \gamma)$. The fundamental objective is to learn a policy $\pi$ that maximizes the expected discounted return. Central to this are the Bellman equations, which recursively decompose the value of a state or state-action pair into immediate reward and the discounted value of the subsequent state. Temporal Difference (TD) learning methods, such as Q-learning, bootstrap these estimates without requiring a complete model of the environment dynamics.

## 2. Evolution of Value-Based Methods
The breakthrough in Deep RL occurred with the introduction of the Deep Q-Network (DQN) by Mnih et al. (2013, 2015). DQN stabilized the training of deep neural networks in RL by introducing Experience Replay and Target Networks.

Subsequent research addressed DQN's limitations:
- **Double DQN (DDQN)** (van Hasselt et al., 2016): Addressed the overestimation bias inherent in standard Q-learning by decoupling action selection from action evaluation.
- **Dueling DQN** (Wang et al., 2016): Separated the representation of state value and state-dependent action advantages, improving policy evaluation in states where action choice is irrelevant.
- **Rainbow DQN** (Hessel et al., 2018): Combined multiple independent improvements (DDQN, Dueling, Prioritized Replay, Multi-step learning, Distributional RL, and Noisy Nets) into a single, highly performant agent.

## 3. Evolution of Policy Gradient Methods
Policy gradient methods directly parameterize the policy. 
- **REINFORCE** (Williams, 1992): A foundational Monte Carlo method, but suffers from high variance.
- **Actor-Critic Methods (A2C/A3C)**: Combine policy gradients (Actor) with learned value functions (Critic) acting as baselines to reduce variance.
- **TRPO** (Schulman et al., 2015): Introduced trust region constraints using KL divergence to guarantee monotonic policy improvement, though computationally expensive.
- **PPO** (Schulman et al., 2017): Simplified TRPO by using a clipped surrogate objective function, achieving similar performance with first-order optimization. PPO has become the default algorithm for many continuous and discrete control tasks.

## 4. Deep RL in Games
Games have historically served as the primary testbed for AI.
- **Atari**: The Arcade Learning Environment (ALE) provided the initial benchmark for visual RL.
- **Go**: AlphaGo (Silver et al., 2016) combined deep CNNs, value networks, and Monte Carlo Tree Search to defeat human world champions.
- **Complex Environments**: More recently, DRL has been scaled to handle imperfect information and immense action spaces in games like StarCraft II (AlphaStar) and Dota 2 (OpenAI Five).

## 5. Frame Preprocessing Standardization
Processing raw video frames is computationally demanding. Machado et al. (2018) highlighted the importance of standardized preprocessing for reproducible research in the ALE:
- **Max-pooling** over consecutive frames removes Atari sprite flickering.
- **Frame-skipping** (repeating an action for $k$ frames) speeds up learning by extending the temporal reach of exploration.
- **Frame-stacking** (usually 4 frames) provides the agent with temporal context, allowing it to infer motion (velocity and acceleration) from static images.

## 6. Comparison Table of Methods

| Algorithm | Type | Stability | Sample Efficiency | Output |
|-----------|------|-----------|-------------------|--------|
| Q-Learning| Value-based | Low | Low | Discrete |
| DQN | Value-based | Moderate | Moderate | Discrete |
| REINFORCE | Policy-based| Low | Low | Continuous/Discrete |
| A2C | Actor-Critic| Moderate | Moderate | Continuous/Discrete |
| PPO | Actor-Critic| High | High | Continuous/Discrete |

## References
1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
2. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
3. Machado, M. C., et al. (2018). Revisiting the arcade learning environment.
4. Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning.
