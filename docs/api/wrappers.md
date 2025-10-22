# Wrappers API

Gymnasium wrappers for EldenGym environments.

::: eldengym.wrappers
    options:
      show_source: true
      heading_level: 2

## Available Wrappers

(Documentation will be added as wrappers are implemented)

## Creating Custom Wrappers

You can create custom wrappers using the Gymnasium wrapper API:

```python
import gymnasium as gym
from gymnasium import Wrapper

class CustomWrapper(Wrapper):
    """Custom wrapper example."""

    def __init__(self, env):
        super().__init__(env)
        # Your initialization

    def step(self, action):
        # Modify action or observation
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Custom logic here
        modified_reward = reward * 2.0

        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Custom logic
        return obs, info

# Use the wrapper
env = gym.make("EldenGym-v0")
env = CustomWrapper(env)
```

## Common Wrapper Patterns

### Frame Stacking

```python
from gymnasium.wrappers import FrameStack

env = gym.make("EldenGym-v0")
env = FrameStack(env, num_stack=4)  # Stack last 4 frames
```

### Action Repeat

```python
from gymnasium.wrappers import ActionRepeatWrapper

env = gym.make("EldenGym-v0", frame_skip=1)  # Disable built-in skip
env = ActionRepeatWrapper(env, repeat=4)  # Repeat each action 4 times
```

### Reward Scaling

```python
from gymnasium.wrappers import TransformReward

env = gym.make("EldenGym-v0")
env = TransformReward(env, lambda r: r / 100.0)  # Scale rewards
```

### Frame Resize

```python
from gymnasium.wrappers import ResizeObservation

env = gym.make("EldenGym-v0")
env = ResizeObservation(env, shape=(84, 84))  # Resize to 84x84
```

### Gray Scale

```python
from gymnasium.wrappers import GrayScaleObservation

env = gym.make("EldenGym-v0")
env = GrayScaleObservation(env)  # Convert to grayscale
```

## Combining Wrappers

```python
import gymnasium as gym
from gymnasium.wrappers import (
    ResizeObservation,
    GrayScaleObservation,
    FrameStack,
)

# Create base environment
env = gym.make("EldenGym-v0", scenario_name="margit")

# Apply wrappers in order
env = GrayScaleObservation(env)      # RGB -> Gray
env = ResizeObservation(env, (84, 84))  # Resize
env = FrameStack(env, num_stack=4)   # Stack frames

# Now ready for training
obs, info = env.reset()
print(obs.shape)  # (4, 84, 84) - 4 stacked grayscale frames
```
