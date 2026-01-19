# Wrappers API

Gymnasium wrappers for EldenGym environments.

::: eldengym.wrappers
    options:
      show_source: true
      heading_level: 2

## Available Wrappers

### Dict Observation Wrappers

These wrappers work with EldenGym's Dict observation space (containing 'frame' and memory attributes).

#### DictFrameStack

Stack the last N frames while preserving other observation keys.

```python
from eldengym.wrappers import DictFrameStack

env = eldengym.make("Margit-v0", ...)
env = DictFrameStack(env, num_stack=4)  # Stack last 4 frames
```

#### DictResizeFrame

Resize frames to a target resolution.

```python
from eldengym.wrappers import DictResizeFrame

env = eldengym.make("Margit-v0", ...)
env = DictResizeFrame(env, width=224, height=224)
```

#### DictGrayscaleFrame

Convert frames to grayscale.

```python
from eldengym.wrappers import DictGrayscaleFrame

env = eldengym.make("Margit-v0", ...)
env = DictGrayscaleFrame(env)
```

#### NormalizeMemoryAttributes

Normalize memory attribute values to [0, 1].

```python
from eldengym.wrappers import NormalizeMemoryAttributes

env = eldengym.make("Margit-v0", ...)
env = NormalizeMemoryAttributes(env, attribute_ranges={
    "HeroHp": (0, 1900),
    "NpcHp": (0, 10000),
})
```

### Utility Wrappers

#### HPRefundWrapper

Refund player and/or boss HP after each step. Useful for evaluation and data collection where you want to prevent episode termination due to HP loss.

```python
from eldengym.wrappers import HPRefundWrapper

env = eldengym.make("Margit-v0", ...)

# Refund only player HP (default)
env = HPRefundWrapper(env, refund_player=True, refund_boss=False)

# Refund both player and boss HP
env = HPRefundWrapper(env, refund_player=True, refund_boss=True)
```

**Parameters:**

- `refund_player` (bool): Whether to refund player HP. Default: `True`
- `refund_boss` (bool): Whether to refund boss HP. Default: `False`
- `player_hp_attr` (str): Attribute name for player HP. Default: `'HeroHp'`
- `player_max_hp_attr` (str): Attribute name for player max HP. Default: `'HeroMaxHp'`
- `boss_hp_attr` (str): Attribute name for boss HP. Default: `'NpcHp'`
- `boss_max_hp_attr` (str): Attribute name for boss max HP. Default: `'NpcMaxHp'`

**Info dict additions:**

When `refund_player=True`:
- `info["player_damage_taken"]` - Raw HP damage taken this step
- `info["player_damage_taken_normalized"]` - Damage as fraction of max HP (0.0 to 1.0)

When `refund_boss=True`:
- `info["boss_damage_dealt"]` - Raw HP damage dealt to boss this step
- `info["boss_damage_dealt_normalized"]` - Damage as fraction of boss max HP (0.0 to 1.0)

**Example with reward shaping:**

```python
from eldengym import HPRefundWrapper

env = eldengym.make("Margit-v0", ...)
env = HPRefundWrapper(env, refund_player=True)

obs, info = env.reset()
for _ in range(1000):
    action = policy(obs)
    obs, reward, term, trunc, info = env.step(action)

    # Penalize taking damage
    damage_penalty = -info.get("player_damage_taken_normalized", 0) * 10.0
    shaped_reward = reward + damage_penalty
```

### Legacy Wrappers

These wrappers work with simple array observations (not Dict spaces).

- `FrameStack` - Stack last N frames
- `ResizeFrame` - Resize frames to target shape
- `GrayscaleFrame` - Convert to grayscale

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
