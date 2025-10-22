# Rewards

Understanding and customizing reward functions in EldenGym.

## Default Reward Function

The default reward encourages defeating the boss while staying alive:

```python
def default_reward(obs, info, terminated, truncated):
    reward = 0.0

    # Reward for damaging boss
    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        reward += abs(info['target_hp_delta']) / 10.0

    # Penalty for taking damage
    if 'player_hp_delta' in info and info['player_hp_delta'] < 0:
        reward += info['player_hp_delta'] / 10.0  # Negative value

    # Big bonus for defeating boss
    if terminated and info.get('target_hp', 0) <= 0:
        reward += 100.0

    # Penalty for dying
    if terminated and info.get('player_hp', 0) <= 0:
        reward -= 50.0

    return reward
```

## Custom Reward Functions

### Signature

```python
def my_reward_function(obs, info, terminated, truncated):
    """
    Args:
        obs: Current observation (np.ndarray or dict)
        info: Info dictionary with game state
        terminated: Whether episode ended
        truncated: Whether episode was truncated

    Returns:
        float: Reward value
    """
    reward = 0.0
    # Your logic here
    return reward
```

### Usage

```python
env = gym.make(
    "EldenGym-v0",
    reward_function=my_reward_function
)
```

## Reward Design Examples

### Aggressive Play

Encourage attacking:

```python
def aggressive_reward(obs, info, terminated, truncated):
    reward = 0.0

    # Big reward for boss damage
    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        reward += abs(info['target_hp_delta']) * 2.0  # 2x multiplier

    # Small penalty for taking damage
    if 'player_hp_delta' in info and info['player_hp_delta'] < 0:
        reward += info['player_hp_delta'] * 0.1  # Only 0.1x

    # Huge bonus for winning
    if terminated and info.get('target_hp', 0) <= 0:
        reward += 500.0

    return reward
```

### Defensive Play

Encourage survival:

```python
def defensive_reward(obs, info, terminated, truncated):
    reward = 0.0

    # Reward for staying alive (per step)
    reward += 0.1

    # Moderate reward for boss damage
    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        reward += abs(info['target_hp_delta']) * 0.5

    # Big penalty for taking damage
    if 'player_hp_delta' in info and info['player_hp_delta'] < 0:
        reward += info['player_hp_delta'] * 5.0  # 5x penalty

    # Bonus for winning without damage
    if terminated and info.get('target_hp', 0) <= 0:
        hp_ratio = info['player_hp'] / info['player_max_hp']
        reward += 100.0 * hp_ratio  # Bonus scales with HP remaining

    return reward
```

### Distance-Based

Encourage maintaining optimal distance:

```python
def distance_reward(obs, info, terminated, truncated):
    reward = 0.0

    # Optimal distance: 5-10 units
    distance = info.get('distance', 0)
    if 5 <= distance <= 10:
        reward += 1.0  # Good positioning
    elif distance < 3:
        reward -= 0.5  # Too close
    elif distance > 15:
        reward -= 0.5  # Too far

    # Standard combat rewards
    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        reward += abs(info['target_hp_delta']) / 10.0

    return reward
```

### Time-Based

Encourage speed:

```python
def speed_reward(obs, info, terminated, truncated):
    reward = 0.0

    # Penalty for each step (encourage speed)
    reward -= 0.01

    # Standard damage rewards
    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        reward += abs(info['target_hp_delta']) / 10.0

    # Huge bonus for fast win
    if terminated and info.get('target_hp', 0) <= 0:
        steps = info.get('step_count', 1000)
        reward += 200.0 * (1000 / max(steps, 1))  # Bonus inversely proportional to steps

    return reward
```

### Shaped Reward

Dense rewards for learning:

```python
def shaped_reward(obs, info, terminated, truncated):
    reward = 0.0

    # HP-based shaping
    player_hp_ratio = info['player_hp'] / info['player_max_hp']
    target_hp_ratio = info['target_hp'] / info['target_max_hp']

    # Reward for HP advantage
    reward += (player_hp_ratio - target_hp_ratio) * 0.1

    # Distance shaping
    distance = info.get('distance', 0)
    if distance > 0:
        reward += 1.0 / (distance + 1)  # Closer = better

    # Combat rewards
    if 'target_hp_delta' in info:
        reward += abs(info['target_hp_delta']) * 1.0
    if 'player_hp_delta' in info:
        reward += info['player_hp_delta'] * 2.0

    # Terminal rewards
    if terminated:
        if info.get('target_hp', 0) <= 0:
            reward += 100.0
        else:
            reward -= 50.0

    return reward
```

## Available Info Fields

Use these in your reward function:

```python
info['player_hp']           # Current player HP
info['player_max_hp']       # Max player HP
info['target_hp']           # Current boss HP
info['target_max_hp']       # Max boss HP
info['distance']            # Distance to boss
info['player_animation_id'] # Player animation ID
info['target_animation_id'] # Boss animation ID
info['step_count']          # Steps in current episode

# After step (not in reset):
info['player_hp_delta']     # HP change this step
info['target_hp_delta']     # Boss HP change this step
```

## Reward Scaling

### Normalize Rewards

```python
from gymnasium.wrappers import NormalizeReward

env = gym.make("EldenGym-v0")
env = NormalizeReward(env)  # Normalize rewards using running statistics
```

### Clip Rewards

```python
from gymnasium.wrappers import ClipReward

env = gym.make("EldenGym-v0")
env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
```

### Transform Rewards

```python
from gymnasium.wrappers import TransformReward

env = gym.make("EldenGym-v0")
env = TransformReward(env, lambda r: np.sign(r) * np.log(1 + abs(r)))
```

## Debugging Rewards

### Log Rewards

```python
rewards = []

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)

    if terminated or truncated:
        break

# Analyze
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.title('Rewards over Episode')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.show()

print(f"Total reward: {sum(rewards)}")
print(f"Mean reward: {np.mean(rewards)}")
print(f"Max reward: {max(rewards)}")
print(f"Min reward: {min(rewards)}")
```

### Reward Components

Track individual components:

```python
def tracked_reward(obs, info, terminated, truncated):
    components = {
        'damage_dealt': 0.0,
        'damage_taken': 0.0,
        'survival': 0.0,
        'terminal': 0.0,
    }

    if 'target_hp_delta' in info and info['target_hp_delta'] < 0:
        components['damage_dealt'] = abs(info['target_hp_delta']) / 10.0

    if 'player_hp_delta' in info and info['player_hp_delta'] < 0:
        components['damage_taken'] = info['player_hp_delta'] / 10.0

    components['survival'] = 0.01

    if terminated:
        if info.get('target_hp', 0) <= 0:
            components['terminal'] = 100.0
        else:
            components['terminal'] = -50.0

    # Log components (to TensorBoard, wandb, etc.)
    # ...

    return sum(components.values())

env = gym.make("EldenGym-v0", reward_function=tracked_reward)
```

## Best Practices

1. **Start Simple** - Begin with sparse rewards (win/loss only)
2. **Add Shaping Gradually** - Introduce dense rewards step by step
3. **Balance Components** - Ensure no single component dominates
4. **Test Thoroughly** - Run random policy to check reward distribution
5. **Scale Appropriately** - Keep rewards in reasonable range (-1 to 1 or -100 to 100)
6. **Avoid Reward Hacking** - Test for unintended behaviors

## Next Steps

- [Environment Guide](environment.md)
- [Action Spaces](action-spaces.md)
- [Examples](../examples/random_policy.ipynb)
