# Action Spaces

EldenGym supports three action space types: discrete, multi-binary, and continuous.

## Discrete Actions (Default)

**Type:** `gymnasium.spaces.Discrete(9)`

The simplest action space with 9 discrete actions:

| Action | Description |
|--------|-------------|
| 0 | No-op (do nothing) |
| 1 | Move forward |
| 2 | Move backward |
| 3 | Strafe left |
| 4 | Strafe right |
| 5 | Attack (R1) |
| 6 | Dodge roll |
| 7 | Toggle lock-on |
| 8 | Use item |

### Usage

```python
env = gym.make("EldenGym-v0", action_mode="discrete")

# Sample random action
action = env.action_space.sample()  # Returns int in [0, 8]

# Take specific action
obs, reward, terminated, truncated, info = env.step(5)  # Attack
```

### Best For
- Simple agents
- Q-learning, DQN
- Quick prototyping

## Multi-Binary Actions

**Type:** `gymnasium.spaces.MultiBinary(8)`

Allow simultaneous actions with binary vector:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | Forward | Move forward |
| 1 | Backward | Move backward |
| 2 | Left | Strafe left |
| 3 | Right | Strafe right |
| 4 | Attack | Attack (R1) |
| 5 | Dodge | Dodge roll |
| 6 | Lock-on | Toggle lock-on |
| 7 | Use item | Use item |

### Usage

```python
env = gym.make("EldenGym-v0", action_mode="multi_binary")

# Move forward and attack
action = np.array([1, 0, 0, 0, 1, 0, 0, 0])
obs, reward, terminated, truncated, info = env.step(action)

# Dodge while moving left
action = np.array([0, 0, 1, 0, 0, 1, 0, 0])
obs, reward, terminated, truncated, info = env.step(action)
```

### Best For
- More complex behaviors
- Combining movements with actions
- PPO, A2C agents

## Continuous Actions

**Type:** `gymnasium.spaces.Box(low=-1, high=1, shape=(8,))`

Continuous control for each action:

```python
env = gym.make("EldenGym-v0", action_mode="continuous")

# Values in [-1, 1]
# Threshold determines activation (e.g., > 0.5)
action = np.array([0.8, 0.0, -0.3, 0.0, 0.9, 0.0, 0.0, 0.0])
obs, reward, terminated, truncated, info = env.step(action)
```

### Best For
- Advanced control
- SAC, TD3 agents
- Research purposes

## Custom Action Mappings

Create custom wrappers for different action schemes:

```python
class CustomActionWrapper(gym.ActionWrapper):
    """Map continuous actions to discrete."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        """Convert continuous [move, attack] to discrete."""
        move = action[0]
        attack = action[1]

        if attack > 0.5:
            return 5  # Attack
        elif move > 0.5:
            return 1  # Forward
        elif move < -0.5:
            return 2  # Backward
        else:
            return 0  # No-op
```

## Action Timing

### Frame Skip

Actions are repeated for `frame_skip` frames:

```python
# Action repeated for 4 frames (default)
env = gym.make("EldenGym-v0", frame_skip=4)

# No frame skip
env = gym.make("EldenGym-v0", frame_skip=1)
```

### Hold Time

Key hold duration (in milliseconds):

```python
# Hold keys for 100ms
client.input_key_tap(['W'], hold_ms=100)

# Hold keys for 500ms (longer action)
client.input_key_tap(['R1'], hold_ms=500)
```

## Action Examples

### Defensive Play

```python
# Discrete: dodge frequently
actions = [6, 6, 1, 6, 5]  # Dodge, move, attack

# Multi-binary: dodge while moving
action = np.array([1, 0, 0, 0, 0, 1, 0, 0])  # Forward + dodge
```

### Aggressive Play

```python
# Discrete: attack combo
actions = [5, 5, 5, 6]  # Attack x3, dodge

# Multi-binary: attack while moving
action = np.array([1, 0, 0, 0, 1, 0, 1, 0])  # Forward + attack + lock-on
```

### Kiting Strategy

```python
# Move backward while locked on
action = np.array([0, 1, 0, 0, 0, 0, 1, 0])  # Backward + lock-on
```

## Next Steps

- [Observation Spaces](observation-spaces.md)
- [Rewards](rewards.md)
- [Examples](../examples/random_policy.ipynb)
