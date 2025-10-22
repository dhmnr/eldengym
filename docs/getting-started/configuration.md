# Configuration

Learn how to configure EldenGym for your use case.

## TOML Configuration Files

EldenGym uses TOML files to define game memory patterns and process information.

### Default Configuration

The default config is located at:
```
eldengym/files/configs/ER_1_16_1.toml
```

### Config Structure

```toml
[process_info]
name = "eldenring.exe"
window_name = "ELDEN RING"

[attributes.AttributeName]
pattern = "48 8B 05 ?? ?? ?? ??"
offsets = [0x10EF8, 0x0, 0x190, 0x138]
type = "int"  # or "float", "array"
length = 4    # For array types
method = ""   # Optional
```

## Memory Attributes

### Common Attributes

The default config includes:

**Player Stats:**
- `HeroHp` / `HeroMaxHp` - Health
- `HeroFp` / `HeroMaxFp` - FP (mana)
- `HeroSp` / `HeroMaxSp` - Stamina

**Player Position:**
- `HeroLocalPosX/Y/Z` - Local coordinates
- `HeroGlobalPosX/Y/Z` - Global coordinates

**Player Animation:**
- `HeroAnimId` - Current animation ID
- `HeroAnimLength` - Animation length
- `HeroAnimSpeed` - Animation speed

**Target/Boss:**
- `NpcHp` / `NpcMaxHp` - Target health
- `NpcLocalPosX/Y/Z` - Target position
- `NpcGlobalPosX/Y/Z` - Target global position
- `NpcAnimId` - Target animation

### Pattern Syntax

Memory patterns use AOB (Array of Bytes) format:

```
48 8B 05 ?? ?? ?? ?? 48 85 C0
```

- **Fixed bytes**: `48`, `8B`, `05` (must match exactly)
- **Wildcards**: `??` (matches any byte)

### Offset Chains

Offsets define a pointer chain to follow:

```toml
offsets = [0x10EF8, 0x0, 0x190, 0x138]
```

This means:
1. Start at pattern match
2. Add `0x10EF8`, read pointer
3. Add `0x0`, read pointer
4. Add `0x190`, read pointer
5. Add `0x138` = final address

## Creating Custom Configs

### 1. Find Memory Patterns

Use tools like Cheat Engine or x64dbg to find memory patterns:

```
1. Search for a value (e.g., player HP)
2. Change the value in-game
3. Refine search
4. Find the base pointer
5. Generate AOB pattern
```

### 2. Create Config File

```toml
[process_info]
name = "mygame.exe"
window_name = "My Game Window"

[attributes.PlayerHealth]
pattern = "48 8B 05 ?? ?? ?? ?? 48 85 C0"
offsets = [0x1000, 0x20, 0x10]
type = "int"

[attributes.PlayerMana]
pattern = "48 8B 05 ?? ?? ?? ?? 48 85 C0"
offsets = [0x1000, 0x20, 0x14]
type = "int"
```

### 3. Use Custom Config

```python
env = gym.make(
    "EldenGym-v0",
    config_filepath="/path/to/custom_config.toml"
)
```

## Environment Options

### Scenario Configuration

```python
env = gym.make(
    "EldenGym-v0",
    scenario_name="margit",  # Boss scenario
)
```

### Action Space Configuration

```python
# Discrete actions (default)
env = gym.make("EldenGym-v0", action_mode="discrete")

# Multi-binary (simultaneous actions)
env = gym.make("EldenGym-v0", action_mode="multi_binary")

# Continuous actions
env = gym.make("EldenGym-v0", action_mode="continuous")
```

### Observation Configuration

```python
# RGB frames only (default)
env = gym.make("EldenGym-v0", observation_mode="rgb")

# Dictionary with game state
env = gym.make("EldenGym-v0", observation_mode="dict")
```

### Game Speed Configuration

```python
env = gym.make(
    "EldenGym-v0",
    frame_skip=4,        # Skip 4 frames (like Atari)
    game_speed=1.0,      # Normal speed
    freeze_game=False,   # Don't freeze between steps
    game_fps=60,         # Target 60 FPS
)
```

### Episode Configuration

```python
env = gym.make(
    "EldenGym-v0",
    max_step=1000,  # Max 1000 steps per episode
)
```

## Custom Reward Functions

```python
def my_reward_function(obs, info, terminated, truncated):
    """Custom reward function.

    Args:
        obs: Current observation
        info: Info dictionary
        terminated: Whether episode terminated
        truncated: Whether episode truncated

    Returns:
        float: Reward value
    """
    reward = 0.0

    # Reward for boss damage
    if 'target_hp_delta' in info:
        reward += info['target_hp_delta'] * 10.0

    # Penalty for player damage
    if 'player_hp_delta' in info:
        reward += info['player_hp_delta'] * 5.0

    # Bonus for winning
    if terminated and info.get('target_hp', 0) <= 0:
        reward += 1000.0

    return reward

# Use custom reward
env = gym.make(
    "EldenGym-v0",
    reward_function=my_reward_function
)
```

## Server Configuration

### Connection Settings

```python
from eldengym.client.elden_client import EldenClient

client = EldenClient(
    host="localhost:50051",                  # Server address
    max_receive_message_length=100*1024*1024,  # 100MB
    max_send_message_length=100*1024*1024,     # 100MB
)
```

### Remote Server

```python
# Connect to remote server
client = EldenClient(host="192.168.1.100:50051")
env = gym.make("EldenGym-v0", host="192.168.1.100:50051")
```

## Troubleshooting

### Config Not Found

EldenGym auto-resolves config paths:

```python
# These all work:
config="ER_1_16_1.toml"  # Searches in eldengym/files/configs/
config="files/configs/ER_1_16_1.toml"  # Relative to package
config="/absolute/path/to/config.toml"  # Absolute path
```

### Memory Pattern Outdated

If the game updates, memory patterns may change:

1. Update the config file with new patterns
2. Use pattern + offset combinations that are version-agnostic
3. Check community for updated configs

### Performance Issues

```python
# Reduce frame capture cost
env = gym.make(
    "EldenGym-v0",
    frame_skip=8,        # Skip more frames
    observation_mode="dict",  # Don't capture frames if not needed
)
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [API Reference](../api/env.md)
