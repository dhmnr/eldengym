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

print(f"Initialized: {results['memory']['success']}")
```

### `launch_game()`

Launch Elden Ring game executable.

**Returns:**
- `dict`: Response from execute_command with keys `'success'`, `'message'`, `'process_id'`

## Player Methods

### Properties

All properties now return clean values (not dict responses):

#### `player_hp`
Get player's current HP.

```python
hp = client.player_hp  # Returns: int (e.g., 1317)
print(f"HP: {hp}")
```

#### `player_max_hp`
Get player's maximum HP.

**Returns:** `int`

#### `local_player_coords`
Get player's local coordinates (x, y, z).

```python
x, y, z = client.local_player_coords  # Returns: tuple[float, float, float]
```

#### `global_player_coords`
Get player's global coordinates (x, y, z).

**Returns:** `tuple[float, float, float]`

#### `player_animation_id`
Get current player animation ID.

**Returns:** `int`

### Methods

#### `set_player_hp(hp)`
Set player's HP. Note: `set_attribute` now requires a data type parameter.

```python
client.set_player_hp(1000)  # Set HP to 1000
# Internally calls: set_attribute("HeroHp", 1000, "int")
```

#### `teleport(x, y, z)`
Teleport player to coordinates.

```python
client.teleport(100.0, 200.0, 50.0)
# Internally uses: set_attribute("HeroLocalPosX", value, "float")
```

## Target/Boss Methods

### Properties

All properties now return clean values:

#### `target_hp`
Get target's current HP.

**Returns:** `int` (e.g., 167)

```python
hp = client.target_hp  # Returns: 167 (not a dict)
```

#### `target_max_hp`
Get target's maximum HP.

**Returns:** `int`

#### `local_target_coords`
Get target's local coordinates (x, y, z).

**Returns:** `tuple[float, float, float]`

#### `global_target_coords`
Get target's global coordinates (x, y, z).

**Returns:** `tuple[float, float, float]`

#### `target_animation_id`
Get current target animation ID.

**Returns:** `int`

### Methods

#### `set_target_hp(hp)`
Set target's HP.

```python
client.set_target_hp(500)  # Set boss HP to 500
# Internally calls: set_attribute("NpcHp", 500, "int")
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

**Args:**
- `speed` (float): Game speed multiplier (e.g., 0.5 for half speed, 2.0 for double speed)

```python
client.set_game_speed(2.0)  # 2x speed
client.set_game_speed(0.5)  # Half speed
# Internally uses: set_attribute("gameSpeedFlag", True, "bool")
#                  set_attribute("gameSpeedVal", speed, "float")
```

### `start_scenario(scenario_name)`
Start a boss fight scenario by teleporting to fog wall and entering.

**Args:**
- `scenario_name` (str): Name of scenario from `client.scenarios` dict (e.g., "Margit-v0")

```python
client.start_scenario("Margit-v0")
```

## Save File Management

### `copy_save_file(save_file_name, save_file_dir=None, timeout_seconds=10)`
Copy a backup save file to become the active save file. Useful for resetting to a specific game state.

**Args:**
- `save_file_name` (str): Name of backup save file (e.g., "margit_checkpoint.sl2")
- `save_file_dir` (str, optional): Directory containing save files. Defaults to `%APPDATA%/EldenRing/`
- `timeout_seconds` (int): Timeout for copy operation (default: 10)

**Returns:**
- `dict`: Response from execute_command

**Example:**
```python
save_dir = r"C:\Users\YourName\AppData\Roaming\EldenRing\76561198012345678"
client.copy_save_file("margit_checkpoint.sl2", save_file_dir=save_dir)
```

## Menu Navigation

### `enter_menu()`
Enter the game from title screen.

```python
client.enter_menu()  # Presses ENTER three times with delays
```

### `quit_to_title()`
Quit to title screen from in-game.

```python
client.quit_to_title()  # ESC → navigate menu → confirm quit
```

## Low-Level Methods

These methods are inherited from `SiphonClient` (via `pysiphon`) and provide direct game control:

- `input_key_tap(keys, hold_ms=100, delay_ms=0)` - Send keyboard input
- `move_mouse(delta_x, delta_y, steps=1)` - Move mouse
- `input_key_toggle(key, pressed)` - Press/release key (for persistent input)
- `get_attribute(name)` - Read memory value (returns `dict` with `'value'` key)
- `set_attribute(name, value, value_type)` - Write memory value (requires type: "int", "float", "bool")
- `start_frame_stream(format, quality)` - Start non-blocking frame stream
- `get_latest_frame(handle)` - Poll latest frame from stream
- `execute_command(...)` - Execute system command

**Important:** All memory read/write operations now use dictionary responses:

```python
# Reading attributes (returns dict)
response = client.get_attribute("HeroHp")
hp = response["value"]  # Extract the value

# Writing attributes (requires type)
client.set_attribute("HeroHp", 1000, "int")
client.set_attribute("gameSpeedVal", 2.0, "float")
client.set_attribute("gameSpeedFlag", True, "bool")
```

See [SiphonClient API](siphon-client.md) for details.

## Example: Complete Workflow

```python
from eldengym.client.elden_client import EldenClient
import time

# Create client
client = EldenClient(host="localhost:50051")

# Initialize everything
results = client.load_config_from_file("ER_1_16_1.toml", wait_time=2)

# Get player info (properties now return clean values)
print(f"Player HP: {client.player_hp}/{client.player_max_hp}")
print(f"Boss HP: {client.target_hp}/{client.target_max_hp}")
print(f"Distance: {client.target_player_distance:.2f}")

# Control the game
client.input_key_tap(["W"], 500)  # Move forward for 500ms
client.input_key_tap(["SPACE"], 100)  # Jump

# Start frame streaming
handle = client.start_frame_stream(format="jpeg", quality=85)
time.sleep(0.1)  # Wait for first frame

# Get latest frame
frame_data = client.get_latest_frame(handle)
if frame_data:
    print(f"Frame size: {len(frame_data.data)} bytes")
    print(f"Frame dimensions: {frame_data.width}x{frame_data.height}")

# Stop stream
client.stop_frame_stream(handle)

# Clean up
client.close()
```
