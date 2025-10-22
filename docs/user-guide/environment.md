# Environment Guide

Complete guide to using the EldenGym environment.

## Creating an Environment

```python
import gymnasium as gym
import eldengym

env = gym.make("EldenGym-v0")
```

## Environment Lifecycle

### 1. Reset

Start a new episode:

```python
observation, info = env.reset()

print(f"Player HP: {info['player_hp']}")
print(f"Boss HP: {info['target_hp']}")
print(f"Observation shape: {observation.shape}")
```

### 2. Step

Execute an action:

```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

if terminated:
    print("Episode ended (boss defeated or player died)")
if truncated:
    print("Episode truncated (max steps reached)")
```

### 3. Render

Get the current frame:

```python
frame = env.render()
# Returns RGB numpy array (H, W, 3)
```

### 4. Close

Clean up resources:

```python
env.close()
```

## Complete Example

```python
import gymnasium as gym
import eldengym

# Create environment
env = gym.make("EldenGym-v0", scenario_name="margit")

# Training loop
for episode in range(10):
    obs, info = env.reset()
    episode_reward = 0

    done = False
    while not done:
        # Your agent logic here
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        # Optionally render
        # frame = env.render()

    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

env.close()
```

## Configuration Options

See the [Configuration Guide](../getting-started/configuration.md) for all options.

## Common Patterns

### Fixed Episode Length

```python
env = gym.make("EldenGym-v0", max_step=1000)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Monitoring Stats

```python
obs, info = env.reset()
stats = {
    'player_hp': [],
    'target_hp': [],
    'rewards': []
}

for step in range(1000):
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    stats['player_hp'].append(info['player_hp'])
    stats['target_hp'].append(info['target_hp'])
    stats['rewards'].append(reward)

    if terminated or truncated:
        break

# Plot stats
import matplotlib.pyplot as plt
plt.plot(stats['player_hp'], label='Player HP')
plt.plot(stats['target_hp'], label='Boss HP')
plt.legend()
plt.show()
```

### Custom Reset Logic

```python
class CustomEnv(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Custom initialization
        self.env.unwrapped.client.set_player_hp(9999)
        return obs, info

env = CustomEnv(gym.make("EldenGym-v0"))
```

## Troubleshooting

### Environment Won't Reset

**Problem:** Reset hangs or fails

**Solutions:**
- Ensure Siphon server is running
- Check game is at the correct location
- Verify memory patterns are correct

### Actions Don't Work

**Problem:** Actions have no effect in game

**Solutions:**
- Ensure input subsystem is initialized
- Check game has focus
- Verify key mappings match your game settings

### Poor Performance

**Problem:** Environment runs slowly

**Solutions:**
```python
# Increase frame skip
env = gym.make("EldenGym-v0", frame_skip=8)

# Disable frame capture if not needed
env = gym.make("EldenGym-v0", observation_mode="dict")

# Adjust game speed
env = gym.make("EldenGym-v0", game_speed=2.0)
```

## Next Steps

- [Action Spaces Guide](action-spaces.md)
- [Observation Spaces Guide](observation-spaces.md)
- [Rewards Guide](rewards.md)
