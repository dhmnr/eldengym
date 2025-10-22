# Utilities API

Helper functions and utilities for EldenGym.

## Configuration Parsing

::: eldengym.utils.parse_config_file
    options:
      show_source: true
      heading_level: 3

### Example

```python
from eldengym.utils import parse_config_file

# Parse a TOML config file
process_name, window_name, attributes = parse_config_file(
    "eldengym/files/configs/ER_1_16_1.toml"
)

print(f"Process: {process_name}")
print(f"Window: {window_name}")
print(f"Attributes: {len(attributes)}")

# Inspect an attribute
attr = attributes[0]
print(f"Name: {attr['name']}")
print(f"Type: {attr['type']}")
print(f"Pattern: {attr['pattern']}")
print(f"Offsets: {attr['offsets']}")
```

### Config File Format

TOML configuration files define the game process and memory attributes:

```toml
[process_info]
name = "eldenring.exe"
window_name = "ELDEN RING"

[attributes.HeroHp]
pattern = "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F 48 39 88"
offsets = [0x10EF8, 0x0, 0x190, 0x0, 0x138]
type = "int"

[attributes.HeroMaxHp]
pattern = "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F 48 39 88"
offsets = [0x10EF8, 0x0, 0x190, 0x0, 0x13C]
type = "int"

[attributes.HeroPosX]
pattern = "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F 48 39 88"
offsets = [0x10EF8, 0x0, 0x190, 0x68]
type = "float"

[attributes.CustomArray]
pattern = "some pattern"
offsets = [0x1000, 0x20]
type = "array"
length = 16  # For array types
method = ""   # Optional method
```

### Attribute Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Attribute identifier |
| `pattern` | str | Memory scan pattern (AOB) |
| `offsets` | list[int] | Pointer chain offsets |
| `type` | str | Data type: "int", "float", "array" |
| `length` | int | Length for array types (optional) |
| `method` | str | Access method (optional) |

### Pattern Syntax

Memory patterns use IDA/x64dbg style with `??` for wildcards:

```
48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F
```

- Fixed bytes: `48`, `8B`, `05`
- Wildcards: `??` (matches any byte)

## Return Values

### parse_config_file()

**Returns:**
- `process_name` (str): Name of the target process
- `process_window_name` (str): Window title of the process
- `attributes` (list[dict]): List of memory attributes

Each attribute dict contains:
```python
{
    'name': str,
    'pattern': str,
    'offsets': list[int],
    'type': str,
    'length': int,
    'method': str,
}
```

## Error Handling

```python
from eldengym.utils import parse_config_file

try:
    process_name, window_name, attributes = parse_config_file("config.toml")
except FileNotFoundError:
    print("Config file not found!")
except ValueError as e:
    print(f"Invalid config format: {e}")
```

Possible errors:
- `FileNotFoundError`: Config file doesn't exist
- `ValueError`: Missing required sections or malformed TOML
