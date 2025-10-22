# Observation Spaces

EldenGym provides flexible observation spaces for different use cases.

## RGB Mode (Default)

**Type:** `gymnasium.spaces.Box(0, 255, (H, W, 3), uint8)`

Returns only the game frame as RGB image.

```python
env = gym.make("EldenGym-v0", observation_mode="rgb")

obs, info = env.reset()
print(obs.shape)  # (1080, 1920, 3) - depends on game resolution
```

### Usage

```python
import matplotlib.pyplot as plt

obs, info = env.reset()
plt.imshow(obs)
plt.show()
```

### Best For
- Vision-based agents
- CNN policies
- Pixel-to-action learning

## Dictionary Mode

**Type:** `gymnasium.spaces.Dict({...})`

Returns frame plus game state information.

```python
env = gym.make("EldenGym-v0", observation_mode="dict")

obs, info = env.reset()
print(obs.keys())  # ['frame', 'player_hp', 'target_hp', ...]
```

### Structure

```python
{
    'frame': Box(0, 255, (H, W, 3), uint8),
    'player_hp': Box(0, inf, (1,), float32),
    'player_max_hp': Box(0, inf, (1,), float32),
    'target_hp': Box(0, inf, (1,), float32),
    'target_max_hp': Box(0, inf, (1,), float32),
    'distance': Box(0, inf, (1,), float32),
    'player_animation_id': Box(0, inf, (1,), float32),
    'target_animation_id': Box(0, inf, (1,), float32),
}
```

### Usage

```python
obs, info = env.reset()

frame = obs['frame']
player_hp = obs['player_hp'][0]
boss_hp = obs['target_hp'][0]
distance = obs['distance'][0]

print(f"HP: {player_hp}/{obs['player_max_hp'][0]}")
print(f"Boss HP: {boss_hp}/{obs['target_max_hp'][0]}")
print(f"Distance: {distance:.2f}")
```

### Best For
- Multi-modal agents
- State-based policies
- Hybrid vision + state approaches

## Custom Observations

### Wrapper for Grayscale

```python
from gymnasium.wrappers import GrayScaleObservation

env = gym.make("EldenGym-v0")
env = GrayScaleObservation(env)

obs, info = env.reset()
print(obs.shape)  # (H, W) - grayscale
```

### Wrapper for Resize

```python
from gymnasium.wrappers import ResizeObservation

env = gym.make("EldenGym-v0")
env = ResizeObservation(env, shape=(84, 84))

obs, info = env.reset()
print(obs.shape)  # (84, 84, 3)
```

### Wrapper for Frame Stacking

```python
from gymnasium.wrappers import FrameStack

env = gym.make("EldenGym-v0")
env = FrameStack(env, num_stack=4)

obs, info = env.reset()
print(obs.shape)  # (4, H, W, 3) - 4 stacked frames
```

### Combined Preprocessing

```python
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)

# Create Atari-style preprocessing
env = gym.make("EldenGym-v0", frame_skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStack(env, num_stack=4)

obs, info = env.reset()
print(obs.shape)  # (4, 84, 84) - ready for DQN/PPO
```

## Custom Observation Wrapper

```python
import gymnasium as gym
import numpy as np

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observation values."""

    def __init__(self, env):
        super().__init__(env)
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """Normalize to [0, 1]."""
        return obs.astype(np.float32) / 255.0

# Use it
env = NormalizeObservation(gym.make("EldenGym-v0"))
```

## State-Only Observation

If you don't need frames:

```python
class StateOnlyWrapper(gym.ObservationWrapper):
    """Return only game state, no frames."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )

    def observation(self, obs):
        """Extract state from info."""
        info = self.env.unwrapped.last_info
        return np.array([
            info['player_hp'],
            info['player_max_hp'],
            info['target_hp'],
            info['target_max_hp'],
            info['distance'],
            info['player_animation_id'],
            info['target_animation_id'],
        ], dtype=np.float32)

env = StateOnlyWrapper(gym.make("EldenGym-v0"))
```

## Performance Considerations

### Reduce Frame Cost

```python
# Don't capture frames if not needed
env = gym.make("EldenGym-v0", observation_mode="dict")

# Only use state
obs = {k: v for k, v in obs.items() if k != 'frame'}
```

### Reduce Resolution

```python
# Lower resolution = faster processing
env = gym.make("EldenGym-v0")
env = ResizeObservation(env, (320, 180))  # Much smaller
```

### Skip Frames

```python
# Higher frame skip = fewer observations
env = gym.make("EldenGym-v0", frame_skip=8)
```

## Info Dictionary

Additional information available in `info`:

```python
obs, info = env.reset()

# Available keys:
info['player_hp']           # Current HP
info['player_max_hp']       # Max HP
info['target_hp']           # Boss HP
info['target_max_hp']       # Boss max HP
info['distance']            # Distance to boss
info['player_animation_id'] # Player animation
info['target_animation_id'] # Boss animation
info['step_count']          # Steps in episode

# After step:
obs, reward, terminated, truncated, info = env.step(action)
info['player_hp_delta']     # HP change
info['target_hp_delta']     # Boss HP change
```

## Next Steps

- [Action Spaces](action-spaces.md)
- [Rewards](rewards.md)
- [API Reference](../api/env.md)
