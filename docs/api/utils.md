# Utilities API

Helper functions and utilities for EldenGym.

## File Path Resolution

::: eldengym.utils.resolve_file_path
    options:
      show_source: true
      heading_level: 3

### Example

```python
from eldengym.utils import resolve_file_path

# Resolve a file path relative to the package
config_path = resolve_file_path("files/Margit-v0/er_siphon_config.toml")
print(f"Resolved path: {config_path}")

# Works with absolute paths too
abs_path = resolve_file_path("/absolute/path/to/config.toml")
print(f"Absolute path: {abs_path}")
```

This utility ensures file paths work correctly whether:
- EldenGym is installed as a package
- Running from the project root
- Using relative or absolute paths

### Use Cases

**Loading configuration files:**
```python
from eldengym.utils import resolve_file_path

# Resolve keybinds file
keybinds_path = resolve_file_path("files/Margit-v0/keybinds.json")

# Load the file
with open(keybinds_path, 'r') as f:
    keybinds = json.load(f)
```

**Finding scenario files:**
```python
from eldengym.utils import resolve_file_path

# Get scenario directory
scenario_dir = resolve_file_path("files/Margit-v0")

# List all files in the scenario
for file in scenario_dir.glob("*"):
    print(file.name)
```

## Configuration Files

EldenGym uses TOML configuration files for the Siphon server. These are handled by `pysiphon`:

**File Structure:**
```
eldengym/files/
├── Margit-v0/
│   ├── er_siphon_config.toml  # Siphon memory configuration
│   └── keybinds.json           # Key bindings
└── (other scenarios)
```

**Loading configs:**
```python
import eldengym

# Configs are automatically resolved when using make()
env = eldengym.make("Margit-v0")

# Or manually with EldenClient
from eldengym.client import EldenClient

client = EldenClient()
client.load_config_from_file("files/Margit-v0/er_siphon_config.toml")
```

**Note:** Configuration parsing is now handled by `pysiphon.SiphonClient.set_process_config()`, which reads TOML files directly. EldenGym no longer parses TOML files internally.
