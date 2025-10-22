# Environment API

The main `EldenGymEnv` class implements the Gymnasium environment interface.

## EldenGymEnv

::: eldengym.env.EldenGymEnv
    options:
      show_source: true
      heading_level: 3

## Methods

### Core Gymnasium Methods

#### `reset()`
Reset the environment to initial state.

**Returns:**
- `observation` (np.ndarray): Initial observation
- `info` (dict): Additional information

**Example:**
```python
obs, info = env.reset()
print(f"Starting HP: {info['player_hp']}")
```

#### `step(action)`
Execute one step in the environment.

**Args:**
- `action` (int | np.ndarray): Action to take

**Returns:**
- `observation` (np.ndarray): New observation
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode ended (boss defeated/player died)
- `truncated` (bool): Whether episode was truncated (max steps)
- `info` (dict): Additional information

**Example:**
```python
obs, reward, terminated, truncated, info = env.step(action)
if terminated:
    print(f"Episode ended! Final reward: {reward}")
```

#### `close()`
Clean up environment resources.

```python
env.close()
```

### Rendering

#### `render()`
Return current game frame.

**Returns:**
- `np.ndarray`: RGB frame (H, W, 3)

**Example:**
```python
import matplotlib.pyplot as plt

frame = env.render()
plt.imshow(frame)
plt.show()
```

## Properties

### Action Space

The action space depends on the `action_mode` parameter:

**Discrete (default):**
```python
env.action_space  # Discrete(9)
# 0: no-op
# 1: forward
# 2: backward
# 3: left
# 4: right
# 5: attack
# 6: dodge
# 7: lock-on
# 8: use-item
```

**Multi-Binary:**
```python
env.action_space  # MultiBinary(8)
# [forward, backward, left, right, attack, dodge, lock-on, use-item]
```

### Observation Space

**RGB Frame only:**
```python
env.observation_space  # Box(0, 255, (H, W, 3), uint8)
```

**With game state:**
```python
env.observation_space  # Dict({
#   'frame': Box(0, 255, (H, W, 3), uint8),
#   'player_hp': Box(0, inf, (1,), float32),
#   'player_max_hp': Box(0, inf, (1,), float32),
#   'target_hp': Box(0, inf, (1,), float32),
#   'target_max_hp': Box(0, inf, (1,), float32),
#   ...
# })
```

## Info Dictionary

The `info` dict returned by `step()` and `reset()` contains:

| Key | Type | Description |
|-----|------|-------------|
| `player_hp` | int | Player's current HP |
| `player_max_hp` | int | Player's maximum HP |
| `target_hp` | int | Target/boss current HP |
| `target_max_hp` | int | Target/boss maximum HP |
| `distance` | float | Distance to target |
| `player_animation_id` | int | Current player animation |
| `target_animation_id` | int | Current target animation |
| `step_count` | int | Steps in current episode |

## Configuration

```python
env = gym.make(
    "EldenGym-v0",

    # Scenario
    scenario_name="margit",  # Boss fight scenario

    # Connection
    host="localhost:50051",  # Siphon server address
    config_filepath="ER_1_16_1.toml",  # Memory config

    # Action space
    action_mode="discrete",  # "discrete", "multi_binary", or "continuous"

    # Observation space
    observation_mode="rgb",  # "rgb" or "dict"

    # Rewards
    reward_function=None,  # Custom reward function

    # Game settings
    frame_skip=4,  # Frames to skip (like Atari)
    game_speed=1.0,  # Game speed multiplier
    freeze_game=False,  # Freeze game between steps
    game_fps=60,  # Target FPS

    # Episode settings
    max_step=None,  # Max steps before truncation (None = no limit)
)
```
