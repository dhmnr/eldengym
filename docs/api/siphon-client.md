# SiphonClient API

The `SiphonClient` class provides low-level gRPC communication with the Siphon server.

!!! info "Using pysiphon"
    EldenGym now uses the official [pysiphon](https://pysiphon.dhmnr.sh/) package for Siphon communication. For complete API documentation, see the [pysiphon API reference](https://pysiphon.dhmnr.sh/api/client/).

!!! note "Return Types"
    All pysiphon methods return dictionaries, not protobuf response objects. Access response fields using dictionary keys (e.g., `response['success']`) instead of attributes (e.g., `response.success`).

## Core Methods

### Input Control

#### `input_key_tap(keys, hold_ms, delay_ms=0)`

Send keyboard input to the game.

**Args:**
- `keys` (list[str]): Keys to press (e.g., `['W', 'SPACE']`)
- `hold_ms` (int): Time to hold keys in milliseconds
- `delay_ms` (int): Delay between keys in milliseconds

**Example:**
```python
# Move forward for 500ms
client.input_key_tap(['W'], 500)

# Jump (quick tap)
client.input_key_tap(['SPACE'], 100)

# Multiple keys with delay
client.input_key_tap(['W', 'SHIFT', 'SPACE'], 200, 100)
```

#### `input_key_toggle(key, toggle)`

Press or release a key.

**Args:**
- `key` (str): Key to toggle
- `toggle` (bool): True to press, False to release

```python
# Hold forward
client.input_key_toggle('W', True)
# ... do something ...
client.input_key_toggle('W', False)
```

#### `move_mouse(delta_x, delta_y, steps=1)`

Move the mouse cursor.

**Args:**
- `delta_x` (int): Horizontal movement
- `delta_y` (int): Vertical movement
- `steps` (int): Number of steps to interpolate

```python
# Look right
client.move_mouse(100, 0)

# Look down
client.move_mouse(0, 50)
```

### Memory Operations

#### `get_attribute(attribute_name)`

Read a memory value.

**Args:**
- `attribute_name` (str): Name of the attribute (from config)

**Returns:**
- `dict`: Response with keys:
  - `success` (bool): Whether read was successful
  - `message` (str): Status message
  - `value` (int | float | bytes): The actual attribute value
  - `value_type` (str): Type of the value ('int', 'float', 'bytes')

```python
response = client.get_attribute("HeroHp")
hp = response['value']  # Extract the value

# Or with error checking
response = client.get_attribute("HeroMaxHp")
if response['success']:
    max_hp = response['value']
    print(f"HP: {hp}/{max_hp}")
```

#### `set_attribute(attribute_name, value)`

Write a memory value.

**Args:**
- `attribute_name` (str): Name of the attribute
- `value` (int | float | bytes): Value to write

```python
# Set player HP
client.set_attribute("HeroHp", 1000)

# Set game speed
client.set_attribute("gameSpeedVal", 2.0)
```

### Frame Capture

#### `capture_frame()`

Capture the current game frame.

**Returns:**
- `np.ndarray`: BGR frame (H, W, 3), uint8

```python
frame = client.capture_frame()
print(f"Frame shape: {frame.shape}")  # e.g., (1080, 1920, 3)
```

### Initialization

#### `set_process_config(config_file_path)`

Load TOML config and send to server.

**Args:**
- `config_file_path` (str): Path to TOML configuration file

**Returns:**
- `dict`: Contains keys `success` (bool), `message` (str), and config details

```python
# Point to your TOML config file
client.set_process_config("path/to/config.toml")
```

**Example TOML config file:**
```toml
process_name = "eldenring.exe"
process_window_name = "ELDEN RING"

[[attributes]]
name = "HeroHp"
pattern = "48 8B 05 ?? ?? ?? ??"
offsets = [0x10EF8, 0x0, 0x190]
type = "int"
length = 4
method = ""
```

#### `initialize_memory()`

Initialize the memory subsystem.

**Returns:**
- `dict`: Contains keys `success` (bool), `message` (str), `process_id` (int)

#### `initialize_input(window_name="")`

Initialize the input subsystem.

**Args:**
- `window_name` (str): Target window name (optional)

**Returns:**
- `dict`: Contains keys `success` (bool), `message` (str)

#### `initialize_capture(window_name="")`

Initialize the capture subsystem.

**Args:**
- `window_name` (str): Target window name (optional)

**Returns:**
- `dict`: Contains keys `success` (bool), `message` (str), `window_width` (int), `window_height` (int)

#### `get_server_status()`

Get current server initialization status.

**Returns:**
- `dict`: Server state information with keys like `memory_initialized`, `process_id`, etc.

```python
status = client.get_server_status()
print(f"Memory initialized: {status['memory_initialized']}")
print(f"Process ID: {status['process_id']}")
```

### System Commands

#### `execute_command(command, args=None, working_directory="", timeout_seconds=30, capture_output=True)`

Execute a system command on the server.

**Args:**
- `command` (str): Command to execute
- `args` (list[str]): Command arguments
- `working_directory` (str): Working directory
- `timeout_seconds` (int): Command timeout
- `capture_output` (bool): Whether to capture output

**Returns:**
- `dict`: Contains keys `success` (bool), `message` (str), `exit_code` (int), `stdout_output` (str), `stderr_output` (str)

```python
# Start the game
response = client.execute_command(
    "eldenring.exe",
    working_directory="C:/Games/Elden Ring"
)
print(f"Exit code: {response['exit_code']}")
```

### Connection

#### `close()`

Close the gRPC connection.

```python
client.close()
```

## Connection Parameters

```python
from pysiphon import SiphonClient

client = SiphonClient(
    host="localhost:50051",              # Server address
    max_receive_message_length=100*1024*1024,  # 100MB
    max_send_message_length=100*1024*1024,     # 100MB
)
```

## Usage Notes

!!! note "EldenClient vs SiphonClient"
    For Elden Ring development, use `EldenClient` which inherits from `SiphonClient` and provides game-specific helpers. Use `SiphonClient` directly only for non-Elden Ring applications.

!!! warning "Memory Operations"
    Direct memory operations (`get_attribute`, `set_attribute`) require proper initialization. Always call initialization methods first.
