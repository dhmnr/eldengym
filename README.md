<!-- # EldenGym 🎮

A Gymnasium-compatible reinforcement learning environment for Elden Ring using Siphon memory reading.

## Features

- 🎯 **Gymnasium API**: Standard RL interface following OpenAI Gym conventions
- 🎮 **Action Modes**: Choose between discrete actions or multi-binary key inputs
- ⚡ **Performance**: Configurable frame skip and game speed for faster training
- 🏆 **Boss Scenarios**: Pre-configured environments for common boss fights
- 🔧 **Customizable**: Easy to extend with custom reward functions and wrappers
- 🚀 **Simple Setup**: One-line environment creation with `eldengym.make()`

## Installation

```bash
pip install -e .
```

## Quick Start

### Using the Make Interface (Recommended)

```python
import eldengym

# List available environments
print(eldengym.list_envs())

# Create an environment
env = eldengym.make('EldenRing-Margit-v0')

# Standard Gymnasium loop
observation, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Available Environments

| Environment ID | Description |
|---------------|-------------|
| `EldenRing-Margit-v0` | Margit boss fight, discrete actions, normal speed |
| `EldenRing-Margit-Easy-v0` | Margit boss fight, discrete actions, 0.5x speed |
| `EldenRing-Margit-MultiBinary-v0` | Margit boss fight, multi-binary actions |
| `EldenRing-Godrick-v0` | Godrick boss fight, discrete actions, normal speed |
| `EldenRing-Godrick-Easy-v0` | Godrick boss fight, discrete actions, 0.5x speed |
| `EldenRing-Godrick-MultiBinary-v0` | Godrick boss fight, multi-binary actions |
| `EldenRing-v0` | Generic environment (specify scenario_name) |

### Customizing Parameters

Override default parameters when creating environments:

```python
env = eldengym.make(
    'EldenRing-Margit-v0',
    game_speed=0.7,      # Slow down game
    frame_skip=8,        # Skip more frames
    max_step=1000,       # Episode length limit
    host='localhost:50051'
)
```

### Registering Custom Environments

Create your own pre-configured environments:

```python
eldengym.register(
    id='EldenRing-Margit-Training-v0',
    entry_point=eldengym.EldenGymEnv,
    kwargs={
        'scenario_name': 'margit',
        'action_mode': 'discrete',
        'frame_skip': 8,
        'game_speed': 0.6,
    }
)

env = eldengym.make('EldenRing-Margit-Training-v0')
```

## Action Spaces

### Discrete Mode (14 actions)

```python
env = eldengym.make('EldenRing-Margit-v0')  # action_mode='discrete'
```

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | no-op | Do nothing |
| 1 | forward | Move forward |
| 2 | backward | Move backward |
| 3 | left | Move left |
| 4 | right | Move right |
| 5 | jump | Jump |
| 6 | dodge_forward | Dodge roll forward |
| 7 | dodge_backward | Dodge roll backward |
| 8 | dodge_left | Dodge roll left |
| 9 | dodge_right | Dodge roll right |
| 10 | interact | Interact |
| 11 | attack | Heavy attack |
| 12 | use_item | Use item (heal) |
| 13 | weapon_art | Weapon art/skill |

### Multi-Binary Mode (11 keys)

```python
env = eldengym.make('EldenRing-Margit-MultiBinary-v0')  # action_mode='multi_binary'
```

Press multiple keys simultaneously:

| Index | Key | Description |
|-------|-----|-------------|
| 0 | W | Forward |
| 1 | A | Left |
| 2 | S | Backward |
| 3 | D | Right |
| 4 | SPACE | Jump |
| 5 | LEFT_SHIFT | Dodge/Sprint |
| 6 | E | Interact |
| 7 | LEFT_ALT | Heavy attack modifier |
| 8 | R | Use item |
| 9 | F | Weapon art |
| 10 | LEFT | Attack key |

Example:
```python
action = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # Press W + LEFT_SHIFT = dodge forward
```

## Observations

Each observation includes:

```python
{
    'frame': np.ndarray,           # Game screenshot (RGB)
    'player_hp': float,            # Player health (0.0-1.0)
    'boss_hp': float,              # Boss health (0.0-1.0)
    'distance': float,             # Distance to boss
    'player_animation_id': int,    # Current player animation
    'boss_animation_id': int,      # Current boss animation
    'last_animation_id': int,      # Previous player animation
}
```

## Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario_name` | str | `'margit'` | Boss scenario to load |
| `host` | str | `'localhost:50051'` | Siphon server address |
| `action_mode` | str | `'discrete'` | Action space type |
| `frame_skip` | int | `4` | Frames to skip between steps |
| `game_speed` | float | `1.0` | Game speed multiplier (0.1-1.0) |
| `freeze_game` | bool | `False` | Freeze between observations |
| `game_fps` | int | `60` | Game FPS |
| `max_step` | int | `None` | Maximum steps per episode |
| `reward_function` | RewardFunction | `ScoreDeltaReward` | Custom reward function |

## Custom Reward Functions

Create custom reward functions by inheriting from `RewardFunction`:

```python
from eldengym.rewards import RewardFunction

class MyReward(RewardFunction):
    def calculate(self, obs, info, prev_info):
        # Your reward logic
        if prev_info is None:
            return 0.0

        boss_damage = prev_info['boss_hp'] - info['boss_hp']
        player_damage = prev_info['player_hp'] - info['player_hp']

        return boss_damage * 10 - player_damage * 5

    def is_done(self, obs, info):
        return info['player_hp'] <= 0 or info['boss_hp'] <= 0

env = eldengym.make('EldenRing-Margit-v0', reward_function=MyReward())
```

## Wrappers

EldenGym includes common preprocessing wrappers:

```python
from eldengym.wrappers import GrayscaleFrame, ResizeFrame, FrameStack

env = eldengym.make('EldenRing-Margit-v0')
env = GrayscaleFrame(env)
env = ResizeFrame(env, (84, 84))
env = FrameStack(env, 4)
```

## Direct Instantiation (Advanced)

For more control, you can still directly instantiate the environment:

```python
from eldengym.env import EldenGymEnv

env = EldenGymEnv(
    scenario_name='margit',
    action_mode='discrete',
    frame_skip=4,
    game_speed=1.0,
    max_step=1000
)
```

## Examples

Check out the `examples/` directory:

- `random_policy.ipynb` - Simple random policy example
- `llm_agent.ipynb` - LLM-based agent example
- `using_make_interface.ipynb` - Complete guide to the make interface

## Requirements

- Elden Ring (PC version)
- Siphon memory reading server running
- Python 3.8+
- Dependencies: `gymnasium`, `numpy`, `grpcio`, `protobuf`

## Development

```bash
# Install in development mode
pip install -e .

# Run examples
jupyter notebook examples/
```

## API Comparison

### Old Way (Still Works)
```python
from eldengym.env import EldenGymEnv
env = EldenGymEnv(scenario_name='margit', action_mode='discrete')
```

### New Way (Recommended)
```python
import eldengym
env = eldengym.make('EldenRing-Margit-v0')
```

The new `make` interface:
- ✅ More concise for common configurations
- ✅ Follows Gymnasium conventions
- ✅ Easy to share and reproduce experiments
- ✅ Still allows full parameter customization

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.
 -->
