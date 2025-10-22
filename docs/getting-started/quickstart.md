# Quick Start Guide

This guide will walk you through creating your first RL agent with EldenGym.

## Basic Setup

```python
import gymnasium as gym
import eldengym

# Create environment
env = gym.make("EldenGym-v0", scenario_name="margit")
```

## Simple Random Agent

```python
# Reset environment
observation, info = env.reset()

# Run for 100 steps
for step in range(100):
    # Sample random action
    action = env.action_space.sample()

    # Take action
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}: Reward={reward:.2f}, HP={info.get('player_hp', 0)}")

    # Reset if episode ends
    if terminated or truncated:
        observation, info = env.reset()
        print("Episode ended - resetting")

env.close()
```

## With Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import eldengym

# Create vectorized environment
env = DummyVecEnv([lambda: gym.make("EldenGym-v0", scenario_name="margit")])

# Initialize PPO agent
model = PPO(
    "CnnPolicy",  # Use CNN for image observations
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the model
model.save("margit_ppo")

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## Custom Reward Function

```python
def custom_reward(obs, info, terminated, truncated):
    """Reward function that encourages aggressive play."""
    reward = 0.0

    # Reward for damaging the boss
    if 'target_hp_delta' in info:
        reward += info['target_hp_delta'] * 10.0

    # Penalty for taking damage
    if 'player_hp_delta' in info:
        reward += info['player_hp_delta'] * 5.0

    # Big bonus for winning
    if terminated and info.get('target_hp', 0) <= 0:
        reward += 1000.0

    # Penalty for dying
    if terminated and info.get('player_hp', 0) <= 0:
        reward -= 500.0

    return reward

# Use custom reward
env = gym.make(
    "EldenGym-v0",
    scenario_name="margit",
    reward_function=custom_reward
)
```

## Different Action Spaces

### Discrete Actions (Default)
```python
env = gym.make("EldenGym-v0", action_mode="discrete")
# 9 actions: no-op, forward, backward, left, right, attack, dodge, lock-on, use-item
```

### Multi-Binary Actions
```python
env = gym.make("EldenGym-v0", action_mode="multi_binary")
# Binary vector: [forward, backward, left, right, attack, dodge, lock-on, use-item]
# Can combine actions: [1, 0, 1, 0, 1, 0, 0, 0] = forward + left + attack
```

### Continuous Actions
```python
env = gym.make("EldenGym-v0", action_mode="continuous")
# Continuous control (advanced use)
```

## Environment Options

```python
env = gym.make(
    "EldenGym-v0",
    scenario_name="margit",           # Boss scenario
    action_mode="discrete",           # Action space type
    frame_skip=4,                     # Skip frames (like Atari)
    game_speed=1.0,                   # Game speed multiplier
    max_step=1000,                    # Max steps per episode
    config_filepath="ER_1_16_1.toml", # Game config
)
```

## Monitoring Training

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Save model checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="margit_model"
)

# Evaluate periodically
eval_env = gym.make("EldenGym-v0", scenario_name="margit")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,
)

# Train with callbacks
model.learn(
    total_timesteps=500_000,
    callback=[checkpoint_callback, eval_callback]
)
```

## Next Steps

- Explore [Action Spaces](../user-guide/action-spaces.md)
- Learn about [Observations](../user-guide/observation-spaces.md)
- See [Examples](../examples/random_policy.ipynb)
