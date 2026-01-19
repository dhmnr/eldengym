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

#### AnimFrameWrapper

Track boss animation ID and elapsed frames since animation changed. Useful for learning animation-based dodge timing.

```python
from eldengym.wrappers import AnimFrameWrapper

env = eldengym.make("Margit-v0", ...)
env = AnimFrameWrapper(env)
```

**Parameters:**

- `anim_id_key` (str): Key for animation ID in obs. Default: `'NpcAnimId'`

**Observation additions:**

- `obs["boss_anim_id"]` - Current boss animation ID
- `obs["elapsed_frames"]` - Number of frames since animation changed

#### SDFObsWrapper

Add Signed Distance Field (SDF) observations for arena boundary awareness. Requires an `ArenaBoundary` instance and player coordinates in obs (`player_x`, `player_y` from EldenGymEnv).

```python
from eldengym import SDFObsWrapper, ArenaBoundary

boundary = ArenaBoundary.load("arena_boundary.json")
env = eldengym.make("Margit-v0", ...)
env = SDFObsWrapper(env, boundary=boundary, live_plot=True)
```

**Parameters:**

- `boundary` (ArenaBoundary): Arena boundary instance with `query_sdf(x, y)` method
- `live_plot` (bool): Enable live visualization of positions and SDF. Default: `False`

**Observation additions:**

- `obs["sdf_value"]` - Signed distance to boundary (negative = inside arena)
- `obs["sdf_normal_x"]` - X component of normal vector pointing to boundary
- `obs["sdf_normal_y"]` - Y component of normal vector pointing to boundary

#### OOBSafetyWrapper

Out-of-bounds detection and recovery via teleportation. Uses a soft/hard boundary system where the soft boundary tracks last safe position and the hard boundary triggers teleport.

```python
from eldengym import OOBSafetyWrapper, ArenaBoundary

boundary = ArenaBoundary.load("arena_boundary.json")
env = eldengym.make("Margit-v0", ...)
env = OOBSafetyWrapper(env, boundary=boundary, soft_margin=3.0, hard_margin=0.0)
```

**Parameters:**

- `boundary` (ArenaBoundary): Arena boundary instance
- `soft_margin` (float): Distance inside the hard boundary for safe zone. Default: `3.0`
- `hard_margin` (float): Distance to extend/shrink hard boundary. Default: `0.0`
  - Positive values extend the boundary outward (more permissive)
  - Negative values shrink the boundary inward (more restrictive)

**Boundary thresholds:**

- Hard boundary: `sdf_value < hard_margin`
- Soft boundary: `sdf_value < (hard_margin - soft_margin)`

**Info dict additions:**

- `info["oob_detected"]` - True if player crossed hard boundary
- `info["teleported"]` - True if teleport was triggered
- `info["last_safe_xyz"]` - Last known safe position (inside soft boundary)

**Example configurations:**

```python
# Default: hard at polygon edge, soft 3 units inside
env = OOBSafetyWrapper(env, boundary, soft_margin=3.0, hard_margin=0.0)

# Hard extended 2 units out, soft still 3 units inside hard
env = OOBSafetyWrapper(env, boundary, soft_margin=3.0, hard_margin=2.0)
```

### Combining Wrappers for Dodge Policy

A common pattern for training dodge policies with full wrapper stack:

```python
from eldengym import (
    ArenaBoundary,
    HPRefundWrapper,
    AnimFrameWrapper,
    SDFObsWrapper,
    OOBSafetyWrapper,
)

boundary = ArenaBoundary.load("arena_boundary.json")

env = eldengym.make("Margit-v0", host="192.168.48.1:50051")

# Apply wrappers (order matters!)
env = HPRefundWrapper(env, refund_player=True, refund_boss=False)
env = AnimFrameWrapper(env)
env = SDFObsWrapper(env, boundary=boundary, live_plot=True)
env = OOBSafetyWrapper(env, boundary=boundary, soft_margin=3.0, hard_margin=0.0)

# Now obs contains:
# - player_x, player_y, player_z, boss_x, boss_y, boss_z (from base env)
# - dist_to_boss, boss_z_relative (from base env)
# - boss_anim_id, elapsed_frames (from AnimFrameWrapper)
# - sdf_value, sdf_normal_x, sdf_normal_y (from SDFObsWrapper)
# And info contains damage tracking and OOB detection
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
