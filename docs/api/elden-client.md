# EldenClient API

The `EldenClient` class provides game-specific functionality for Elden Ring.

## EldenClient

::: eldengym.client.elden_client.EldenClient
    options:
      show_source: true
      heading_level: 3

## Initialization Methods

### `load_config_from_file(config_filepath, wait_time=2)`

Complete initialization: load config, initialize memory, input, and capture subsystems.

**Args:**
- `config_filepath` (str | Path): Path to TOML config file
  - Filename only: `"ER_1_16_1.toml"` (searches in `eldengym/files/configs/`)
  - Relative path: `"files/configs/ER_1_16_1.toml"` (from package root)
  - Absolute path: `"/full/path/to/config.toml"`
- `wait_time` (int): Seconds to wait after loading config (default: 2)

**Returns:**
- `dict`: Results with keys `'config'`, `'memory'`, `'input'`, `'capture'`

**Example:**
```python
from eldengym.client.elden_client import EldenClient

client = EldenClient(host="localhost:50051")
results = client.load_config_from_file("ER_1_16_1.toml")

print(f"Initialized: {results['memory'].success}")
```

### `launch_game()`

Launch Elden Ring game executable.

**Returns:**
- `ExecuteCommandResponse`: Command execution result

### `bypass_menu()`

Automatically bypass the main menu to load into the game.

## Player Methods

### Properties

#### `player_hp`
Get player's current HP.

```python
hp = client.player_hp
print(f"HP: {hp}")
```

#### `player_max_hp`
Get player's maximum HP.

#### `local_player_coords`
Get player's local coordinates (x, y, z).

```python
x, y, z = client.local_player_coords
```

#### `global_player_coords`
Get player's global coordinates (x, y, z).

#### `player_animation_id`
Get current player animation ID.

### Methods

#### `set_player_hp(hp)`
Set player's HP.

```python
client.set_player_hp(1000)  # Set HP to 1000
```

#### `teleport(x, y, z)`
Teleport player to coordinates.

```python
client.teleport(100.0, 200.0, 50.0)
```

## Target/Boss Methods

### Properties

#### `target_hp`
Get target's current HP.

#### `target_max_hp`
Get target's maximum HP.

#### `local_target_coords`
Get target's local coordinates (x, y, z).

#### `global_target_coords`
Get target's global coordinates (x, y, z).

#### `target_animation_id`
Get current target animation ID.

### Methods

#### `set_target_hp(hp)`
Set target's HP.

```python
client.set_target_hp(500)  # Set boss HP to 500
```

## Helper Methods

### `target_player_distance`
Get distance between player and target.

```python
distance = client.target_player_distance
print(f"Distance to boss: {distance:.2f}")
```

### `set_game_speed(speed)`
Set game speed multiplier.

```python
client.set_game_speed(2.0)  # 2x speed
client.set_game_speed(0.5)  # Half speed
```

### `reset_game()`
Reset the game (kills player, triggers death/respawn).

```python
client.reset_game()
```

### `start_scenario(scenario_name)`
Start a boss fight scenario.

**Args:**
- `scenario_name` (str): Name of scenario (e.g., "margit")

```python
client.start_scenario("margit")
```

## Low-Level Methods

These methods are inherited from `SiphonClient` and provide direct game control:

- `send_key(keys, hold_time, delay_time)` - Send keyboard input
- `move_mouse(delta_x, delta_y, steps)` - Move mouse
- `toggle_key(key, toggle)` - Press/release key
- `get_attribute(name)` - Read memory value
- `set_attribute(name, value)` - Write memory value
- `get_frame()` - Capture game frame
- `execute_command(...)` - Execute system command

See [SiphonClient API](siphon-client.md) for details.

## Example: Complete Workflow

```python
from eldengym.client.elden_client import EldenClient

# Create client
client = EldenClient(host="localhost:50051")

# Initialize everything
results = client.load_config_from_file("ER_1_16_1.toml", wait_time=2)

# Get player info
print(f"Player HP: {client.player_hp}/{client.player_max_hp}")
print(f"Boss HP: {client.target_hp}/{client.target_max_hp}")
print(f"Distance: {client.target_player_distance:.2f}")

# Control the game
client.send_key(["W"], 500)  # Move forward for 500ms
client.send_key(["SPACE"], 100)  # Jump

# Capture frame
frame = client.get_frame()
print(f"Frame shape: {frame.shape}")

# Clean up
client.close()
```
