# Quick Start Guide

This guide will walk you through creating your first RL agent with EldenGym.

## Basic Setup

```python
import eldengym

# Create environment (use eldengym.make() for registered environments)
env = eldengym.make("Margit-v0", launch_game=False)
```

## Simple Random Agent

```python
# Reset environment
observation, info = env.reset()

# observation is a dict with 'frame' and memory attributes
print(f"Observation keys: {observation.keys()}")
print(f"Frame shape: {observation['frame'].shape}")

# Run for 100 steps
for step in range(100):
    # Sample random action (MultiBinary action space)
    action = env.action_space.sample()

    # Take action
    observation, reward, terminated, truncated, info = env.step(action)

    # Info contains normalized HP values
    hero_hp_pct = info.get('normalized_hero_hp', 0) * 100
    print(f"Step {step}: Reward={reward:.2f}, HP={hero_hp_pct:.1f}%")

    # Reset if episode ends
    if terminated or truncated:
        observation, info = env.reset()
        print("Episode ended - resetting")

env.close()
```

## With Stable-Baselines3

**Note:** You'll need to apply wrappers to flatten the Dict observation space for SB3:

```python
import eldengym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment with wrappers
def make_env():
    env = eldengym.make("Margit-v0", launch_game=False)

    # Apply preprocessing wrappers
    env = eldengym.DictResizeFrame(env, width=84, height=84)
    env = eldengym.DictGrayscaleFrame(env)
    env = eldengym.DictFrameStack(env, num_stack=4)
    env = eldengym.NormalizeMemoryAttributes(env)

    # TODO: Add FlattenDictObservation wrapper for SB3 compatibility
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

# Initialize PPO agent
model = PPO(
    "MultiInputPolicy",  # Use MultiInputPolicy for Dict observations
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
import eldengym
from eldengym.rewards import RewardFunction

class AggressiveReward(RewardFunction):
    """Reward function that encourages aggressive play."""

    def calculate_reward(self, obs, info, prev_info):
        reward = 0.0

        if prev_info is None:
            return 0.0

        # Get current normalized HP values
        hero_hp = info.get('normalized_hero_hp', 0)
        npc_hp = info.get('normalized_npc_hp', 1)

        # Previous normalized HP values
        prev_hero_hp = prev_info.get('normalized_hero_hp', hero_hp)
        prev_npc_hp = prev_info.get('normalized_npc_hp', npc_hp)

        # Reward for damaging the boss (HP delta is negative when damaged)
        npc_damage = prev_npc_hp - npc_hp
        if npc_damage > 0:
            reward += npc_damage * 100.0

        # Penalty for taking damage
        hero_damage = prev_hero_hp - hero_hp
        if hero_damage > 0:
            reward -= hero_damage * 50.0

        return reward

    def check_termination(self, obs, info):
        """End episode when player or boss dies."""
        hero_hp = info.get('normalized_hero_hp', 1)
        npc_hp = info.get('normalized_npc_hp', 1)

        return hero_hp <= 0 or npc_hp <= 0

# Use custom reward
env = eldengym.make(
    "Margit-v0",
    launch_game=False,
    reward_function=AggressiveReward()
)
```

## Action Space (MultiBinary)

The environment uses MultiBinary action space where each element represents a key:

```python
env = eldengym.make("Margit-v0", launch_game=False)

# Check the action keys
print(f"Action keys: {env.action_keys}")
# ['W', 'A', 'S', 'D', 'SPACE', 'E', 'Q', 'R']

# Create action (each element is 0 or 1)
action = [1, 0, 1, 0, 1, 0, 0, 0]  # Press W + S + SPACE simultaneously

# Keys are toggled intelligently - only changed when state differs
obs, reward, terminated, truncated, info = env.step(action)
```

Keys are configured in `keybinds.json` and can be customized per environment.

## Environment Options

```python
env = eldengym.make(
    "Margit-v0",                      # Registered environment
    launch_game=False,                 # Don't launch if game already running
    memory_attributes=[                # Memory values to poll
        "HeroHp", "HeroMaxHp",
        "NpcHp", "NpcMaxHp",
        "HeroAnimId", "NpcAnimId"
    ],
    frame_format="jpeg",               # Frame format ('jpeg' or 'raw')
    frame_quality=85,                  # JPEG quality (1-100)
    max_steps=1000,                    # Max steps per episode
    save_file_name="margit.sl2",       # Backup save to copy on reset
    save_file_dir=r"C:\...\EldenRing\...",  # Save file directory
    reward_function=eldengym.ScoreDeltaReward(),  # Reward calculator
)
```

## Monitoring Training

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import eldengym

# Save model checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="margit_model"
)

# Create evaluation environment
eval_env = eldengym.make("Margit-v0", launch_game=False)
# Apply same wrappers as training env
eval_env = eldengym.DictResizeFrame(eval_env, 84, 84)
eval_env = eldengym.DictGrayscaleFrame(eval_env)

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
