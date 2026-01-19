# Environment API

The main `EldenGymEnv` class implements the Gymnasium environment interface.

## EldenGymEnv

::: eldengym.env.EldenGymEnv
    options:
      show_source: true
      heading_level: 3

## Methods

### Core Gymnasium Methods

#### `reset(seed=None, options=None)`
Reset the environment to initial state.

This method:
1. Releases all keys from previous episode
2. Quits to title screen
3. Copies save file (if configured)
4. Re-enters game and starts scenario

**Args:**
- `seed` (int, optional): Random seed
- `options` (dict, optional): Additional options

**Returns:**
- `observation` (dict): Initial observation with keys `'frame'` and memory attributes
- `info` (dict): Additional information

**Example:**
```python
obs, info = env.reset()
print(f"Frame shape: {obs['frame'].shape}")
print(f"Starting HP: {obs['HeroHp']}")
```

#### `step(action)`
Execute one step in the environment with key toggling.

**Args:**
- `action` (np.ndarray): Multi-binary array where each element is 0 or 1 representing key states

**Returns:**
- `observation` (dict): New observation with `'frame'` and memory attributes
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode ended (determined by reward function)
- `truncated` (bool): Whether episode was truncated (max steps reached)
- `info` (dict): Additional information including normalized values

**Example:**
```python
action = env.action_space.sample()  # Random multi-binary action
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, HP: {obs['HeroHp']}")
if terminated:
    print(f"Episode ended!")
```

#### `close()`
Clean up environment resources.

```python
env.close()
```

### Rendering

#### `render()`
Return current game frame from the latest observation.

**Returns:**
- `np.ndarray`: RGB frame (H, W, 3) in uint8 format

**Example:**
```python
import matplotlib.pyplot as plt

frame = env.render()
plt.imshow(frame)
plt.show()
```

**Note:** The frame is captured asynchronously using `pysiphon`'s frame streaming.

## Properties

### Action Space

The environment uses **MultiBinary** action space where each element represents a key state:

```python
env.action_space  # MultiBinary(n)
# Each element is 0 (key released) or 1 (key pressed)
# The keys are loaded from keybinds.json

# Example with default keybinds:
env.action_keys  # ['W', 'A', 'S', 'D', 'SPACE', 'E', 'Q', 'R']
action = [1, 0, 0, 0, 1, 0, 0, 0]  # Press W and SPACE
```

Keys are toggled intelligently - only changed when the action differs from current state.

### Observation Space

The environment uses **Dict** observation space with frame, memory attributes, and computed real coordinates:

```python
env.observation_space  # Dict({
#   'frame': Box(0, 255, (H, W, 3), uint8),
#
#   # Memory attributes (configurable)
#   'HeroHp': Box(-inf, inf, (), float32),
#   'HeroMaxHp': Box(-inf, inf, (), float32),
#   'NpcHp': Box(-inf, inf, (), float32),
#   'NpcMaxHp': Box(-inf, inf, (), float32),
#   'HeroAnimId': Box(-inf, inf, (), float32),
#   'NpcAnimId': Box(-inf, inf, (), float32),
#
#   # Real coordinates (computed automatically)
#   'player_x': Box(-inf, inf, (), float32),  # Player global X
#   'player_y': Box(-inf, inf, (), float32),  # Player global Y
#   'player_z': Box(-inf, inf, (), float32),  # Player global Z
#   'boss_x': Box(-inf, inf, (), float32),    # Boss global X
#   'boss_y': Box(-inf, inf, (), float32),    # Boss global Y
#   'boss_z': Box(-inf, inf, (), float32),    # Boss global Z
#   'dist_to_boss': Box(0, inf, (), float32), # 2D distance to boss
#   'boss_z_relative': Box(-inf, inf, (), float32),  # Boss Z relative to player
# })
```

The memory attributes are configurable via the `memory_attributes` parameter.

#### Real Coordinates

The environment automatically computes global coordinates for both player and boss using the local-to-global transform:

- **Player coordinates** (`player_x`, `player_y`, `player_z`): Directly from `HeroGlobalPos`
- **Boss coordinates** (`boss_x`, `boss_y`, `boss_z`): Computed as `NpcLocalPos + (HeroGlobalPos - HeroLocalPos)`
- **Distance** (`dist_to_boss`): 2D Euclidean distance (ignores Z)
- **Relative Z** (`boss_z_relative`): `boss_z - player_z` (positive = boss above player)

This transform is necessary because the NPC coordinate system resets at map boundaries, while the Hero global position remains stable.

## Info Dictionary

The `info` dict returned by `step()` and `reset()` contains normalized values:

| Key | Type | Description |
|-----|------|-------------|
| `normalized_hero_hp` | float | Player HP normalized (0-1) |
| `normalized_npc_hp` | float | Boss HP normalized (0-1) |
| Memory attributes | float | Raw values from observation |
| `step_count` | int | Steps in current episode |

**Example:**
```python
obs, info = env.reset()
print(f"Player HP %: {info['normalized_hero_hp'] * 100:.1f}%")
print(f"Boss HP %: {info['normalized_npc_hp'] * 100:.1f}%")
```

## Configuration

### Using `eldengym.make()`

```python
import eldengym

env = eldengym.make(
    "Margit-v0",  # Registered environment

    # Optional overrides:
    launch_game=False,  # Don't launch if already running
    memory_attributes=["HeroHp", "NpcHp", "HeroAnimId"],  # Custom attributes
    frame_format="jpeg",  # Or "raw"
    frame_quality=85,  # JPEG quality (1-100)
    max_steps=1000,  # Max steps per episode

    # Save file management (for reset)
    save_file_name="margit_checkpoint.sl2",
    save_file_dir=r"C:\Users\...\AppData\Roaming\EldenRing\76561198...",

    # Custom reward function
    reward_function=eldengym.ScoreDeltaReward(),
)
```

### Direct Instantiation

```python
from eldengym.env import EldenGymEnv
from eldengym.rewards import ScoreDeltaReward

env = EldenGymEnv(
    scenario_name="Margit-v0",
    keybinds_filepath="path/to/keybinds.json",
    siphon_config_filepath="path/to/er_siphon_config.toml",
    memory_attributes=["HeroHp", "HeroMaxHp", "NpcHp", "NpcMaxHp", "HeroAnimId", "NpcAnimId"],
    host="localhost:50051",
    reward_function=ScoreDeltaReward(),
    frame_format="jpeg",
    frame_quality=85,
    max_steps=None,
    launch_game=True,
    save_file_name=None,
    save_file_dir=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario_name` | str | Required | Boss scenario name |
| `keybinds_filepath` | str | Required | Path to keybinds JSON |
| `siphon_config_filepath` | str | Required | Path to Siphon TOML config |
| `memory_attributes` | list[str] | Default set | Memory values to poll |
| `host` | str | `"localhost:50051"` | Siphon server address |
| `reward_function` | RewardFunction | `ScoreDeltaReward()` | Reward calculator |
| `frame_format` | str | `"jpeg"` | Frame format (`"jpeg"` or `"raw"`) |
| `frame_quality` | int | `85` | JPEG quality (1-100) |
| `max_steps` | int | `None` | Max steps before truncation |
| `launch_game` | bool | `True` | Auto-launch game on init |
| `save_file_name` | str | `None` | Backup save to copy on reset |
| `save_file_dir` | str | `None` | Directory with save files |
