# Methodology

## Environment Description
The primary environment used in this project is `PongNoFrameskip-v4` from the Gymnasium library. 
- **Goal**: Control a paddle to hit a ball past the opponent's paddle.
- **Raw State Space**: RGB images of shape $(210, 160, 3)$.
- **Action Space**: Discrete space of size 6. However, for Pong, only 2 actions are strictly necessary (UP and DOWN) once the game starts. We utilize the full action space to maintain generalizability to other Atari titles.

## Preprocessing Pipeline
Following standard Atari preprocessing protocols, the environment is wrapped sequentially:
1. **NoopReset**: Takes random no-op actions upon reset to ensure random starting states.
2. **MaxAndSkip**: Repeats the selected action for 4 frames and max-pools over the last 2 frames to remove sprite flickering.
3. **EpisodicLife**: Treats end of life as end of episode for value bootstrapping, but only resets the environment on true game over.
4. **WarpFrame**: Grayscales the image and resizes it to $84 \times 84$.
5. **FrameStack**: Stacks the 4 most recent frames along the channel dimension.
Final observation shape: $(4, 84, 84)$.

## Network Architecture
The core perception module is based on Nature CNN:
- Conv2D(in=4, out=32, kernel=8, stride=4, padding=0) + ReLU
- Conv2D(in=32, out=64, kernel=4, stride=2, padding=0) + ReLU
- Conv2D(in=64, out=64, kernel=3, stride=1, padding=0) + ReLU
- Flatten
- Linear(in=3136, out=512) + ReLU

For Q-Networks, the final layer is `Linear(512, num_actions)`.
For Actor-Critic, two heads emerge from the 512-dim layer: a Policy head `Linear(512, num_actions)` and a Value head `Linear(512, 1)`.

## Algorithm Pseudocode Highlights

### PPO (Simplified)
```python
for iteration in range(num_iterations):
    trajectories = collect_data(env, policy_net, num_steps)
    advantages = compute_gae(trajectories, value_net)
    
    for epoch in range(ppo_epochs):
        for batch in get_batches(trajectories):
            ratio = exp(log_prob(batch.actions) - batch.old_log_probs)
            obj1 = ratio * batch.advantages
            obj2 = clip(ratio, 1-eps, 1+eps) * batch.advantages
            policy_loss = -min(obj1, obj2).mean()
            value_loss = MSE(value_net(batch.states), batch.returns)
            
            loss = policy_loss + c1 * value_loss - c2 * entropy
            optimizer.step(loss)
```

## Training Procedure
- Agents are trained for a maximum of 5,000,000 environment frames.
- Models are evaluated periodically (every 50,000 frames) on 10 isolated episodes without exploration noise.
- Checkpoints are saved when the evaluation score improves.

## Evaluation Protocol
The primary metric is the episodic reward. In Pong, scores range from -21 to +21. An agent is considered to have "solved" the environment when it consistently achieves a score of +18 or higher against the built-in AI.
