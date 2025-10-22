# SiphonClient API

The `SiphonClient` class provides low-level gRPC communication with the Siphon server.

## SiphonClient

::: eldengym.client.siphon_client.SiphonClient
    options:
      show_source: true
      heading_level: 3

## Core Methods

### Input Control

#### `send_key(keys, hold_time, delay_time=0)`

Send keyboard input to the game.

**Args:**
- `keys` (list[str]): Keys to press (e.g., `['W', 'SPACE']`)
- `hold_time` (int): Time to hold keys in milliseconds
- `delay_time` (int): Delay between keys in milliseconds

**Example:**
```python
# Move forward for 500ms
client.send_key(['W'], 500)

# Jump (quick tap)
client.send_key(['SPACE'], 100)

# Multiple keys with delay
client.send_key(['W', 'SHIFT', 'SPACE'], 200, 100)
```

#### `toggle_key(key, toggle)`

Press or release a key.

**Args:**
- `key` (str): Key to toggle
- `toggle` (bool): True to press, False to release

```python
# Hold forward
client.toggle_key('W', True)
# ... do something ...
client.toggle_key('W', False)
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
- `int | float | bytes`: The attribute value

```python
hp = client.get_attribute("HeroHp")
max_hp = client.get_attribute("HeroMaxHp")
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

#### `get_frame()`

Capture the current game frame.

**Returns:**
- `np.ndarray`: BGR frame (H, W, 3), uint8

```python
frame = client.get_frame()
print(f"Frame shape: {frame.shape}")  # e.g., (1080, 1920, 3)
```

### Initialization

#### `set_process_config(process_name, process_window_name, attributes)`

Configure the target process and memory attributes.

**Args:**
- `process_name` (str): Process name (e.g., "eldenring.exe")
- `process_window_name` (str): Window name (e.g., "ELDEN RING")
- `attributes` (list[dict]): Memory attribute configurations

```python
attributes = [
    {
        'name': 'HeroHp',
        'pattern': '48 8B 05 ?? ?? ?? ??',
        'offsets': [0x10EF8, 0x0, 0x190],
        'type': 'int',
        'length': 4,
        'method': ''
    }
]

client.set_process_config("eldenring.exe", "ELDEN RING", attributes)
```

#### `initialize_memory()`

Initialize the memory subsystem.

**Returns:**
- `InitializeMemoryResponse`: Contains `success`, `message`, `process_id`

#### `initialize_input(window_name="")`

Initialize the input subsystem.

**Args:**
- `window_name` (str): Target window name (optional)

#### `initialize_capture(window_name="")`

Initialize the capture subsystem.

**Args:**
- `window_name` (str): Target window name (optional)

**Returns:**
- `InitializeCaptureResponse`: Contains `success`, `message`, `window_width`, `window_height`

#### `get_server_status()`

Get current server initialization status.

**Returns:**
- `GetServerStatusResponse`: Server state information

```python
status = client.get_server_status()
print(f"Memory initialized: {status.memory_initialized}")
print(f"Process ID: {status.process_id}")
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
- `ExecuteCommandResponse`: Contains `success`, `message`, `exit_code`, `stdout_output`, `stderr_output`

```python
# Start the game
response = client.execute_command(
    "eldenring.exe",
    working_directory="C:/Games/Elden Ring"
)
print(f"Exit code: {response.exit_code}")
```

### Connection

#### `close()`

Close the gRPC connection.

```python
client.close()
```

## Connection Parameters

```python
from eldengym.client.siphon_client import SiphonClient

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
