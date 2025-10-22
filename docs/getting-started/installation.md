# Installation

## Prerequisites

- **Python 3.10+**
- **Elden Ring** (Steam version)
- **Windows** (required for game interaction)
- **uv** (recommended) or pip

## Install EldenGym

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/dhmnr/eldengym.git
cd eldengym

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/dhmnr/eldengym.git
cd eldengym

# Install in development mode
pip install -e .
```

## Install Siphon Server

EldenGym requires the Siphon gRPC server to communicate with Elden Ring.

1. Download the latest Siphon server release from [releases page]
2. Extract to a convenient location
3. Run `siphon_server.exe`

The server should start on `localhost:50051` by default.

## Verify Installation

```python
import gymnasium as gym
import eldengym

print(eldengym.__version__)  # Should print: 1.0.0

# Test environment creation
env = gym.make("EldenGym-v0")
print("✓ EldenGym installed successfully!")
```

## Configuration Files

EldenGym includes pre-configured memory patterns for Elden Ring:

```
eldengym/
  files/
    configs/
      ER_1_16_1.toml  # Elden Ring v1.16.1
```

If you have a different game version, you may need to update memory patterns.

## Troubleshooting

### "Failed to connect to server"

Make sure the Siphon server is running:
```bash
# Check if server is running
netstat -an | findstr :50051
```

### "Config file not found"

The config path auto-resolves. Use just the filename:
```python
env = gym.make("EldenGym-v0", config_filepath="ER_1_16_1.toml")
```

### "Memory initialization failed"

1. Ensure Elden Ring is running
2. Check game version matches config file
3. Run as Administrator if needed

## Next Steps

Continue to the [Quick Start Guide](quickstart.md) to train your first agent!
