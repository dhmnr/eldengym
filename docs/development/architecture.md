# Architecture

Understanding EldenGym's architecture and design decisions.

## System Overview

```
┌─────────────────────────────────────────────┐
│            RL Training Framework            │
│     (stable-baselines3, Ray RLlib, etc)     │
└────────────────┬────────────────────────────┘
                 │ Gymnasium API
                 ▼
┌─────────────────────────────────────────────┐
│              EldenGym (Python)              │
│  ┌─────────────────────────────────────┐   │
│  │  EldenGymEnv (env.py)               │   │
│  │  - Gymnasium interface              │   │
│  │  - Action/observation spaces        │   │
│  │  - Reward computation               │   │
│  │  - Episode management               │   │
│  └──────────────┬──────────────────────┘   │
│                 │                            │
│  ┌──────────────▼──────────────────────┐   │
│  │  EldenClient (elden_client.py)      │   │
│  │  - Game-specific methods            │   │
│  │  - HP, position, teleport, etc      │   │
│  │  - Config initialization            │   │
│  └──────────────┬──────────────────────┘   │
│                 │                            │
│  ┌──────────────▼──────────────────────┐   │
│  │  SiphonClient (pysiphon package)    │   │
│  │  - Pure gRPC wrapper                │   │
│  │  - Memory operations                │   │
│  │  - Input injection                  │   │
│  │  - Screen capture                   │   │
│  └──────────────┬──────────────────────┘   │
└─────────────────┼────────────────────────────┘
                  │ gRPC (Protocol Buffers)
                  ▼
┌─────────────────────────────────────────────┐
│           Siphon Server (C++)               │
│  ┌─────────────────────────────────────┐   │
│  │  Memory Subsystem                   │   │
│  │  - Pattern scanning                 │   │
│  │  - Pointer dereferencing            │   │
│  │  - Read/write memory                │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Input Subsystem                    │   │
│  │  - Keyboard injection               │   │
│  │  - Mouse injection                  │   │
│  │  - Window focus management          │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Capture Subsystem                  │   │
│  │  - Screen capture (DXGI)            │   │
│  │  - Frame encoding                   │   │
│  │  - Buffer management                │   │
│  └─────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │ Windows API / DirectX
                  ▼
            ┌─────────────┐
            │ Elden Ring  │
            └─────────────┘
```

## Component Details

### 1. EldenGymEnv (env.py)

**Purpose:** Gymnasium environment wrapper

**Responsibilities:**
- Implement Gymnasium API (`reset()`, `step()`, `render()`, `close()`)
- Define action and observation spaces
- Manage episode lifecycle
- Compute rewards
- Handle game state transitions

**Key Methods:**
```python
reset() -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)
render() -> frame
```

### 2. EldenClient (elden_client.py)

**Purpose:** Game-specific high-level interface

**Responsibilities:**
- Game-specific operations (HP, position, etc.)
- Boss fight scenario management
- Config file parsing and initialization
- Convenience methods for common tasks

**Key Features:**
- Properties for game state (`player_hp`, `target_hp`, etc.)
- Helper methods (`teleport()`, `set_game_speed()`)
- Scenario management (`start_scenario()`)
- Auto-resolving config paths

### 3. SiphonClient (pysiphon package)

**Purpose:** Pure gRPC client wrapper

**Responsibilities:**
- Low-level communication with Siphon server
- Direct memory operations
- Input injection
- Frame capture
- System commands

**Design:** Completely game-agnostic, reusable for other games. EldenGym uses the official [pysiphon](https://pysiphon.dhmnr.sh/) package for Siphon communication.

### 4. Siphon Server (C++)

**Purpose:** Low-level game interaction

**Subsystems:**

#### Memory Subsystem
- **Pattern Scanning:** Find memory locations via AOB patterns
- **Pointer Chains:** Follow multi-level pointers
- **Read/Write:** Direct memory access

#### Input Subsystem
- **Keyboard:** Virtual key injection
- **Mouse:** Movement and clicks
- **Focus:** Window focus management

#### Capture Subsystem
- **DXGI:** GPU-accelerated screen capture
- **Encoding:** Frame compression
- **Streaming:** Efficient frame delivery

## Data Flow

### Initialization Flow

```
1. Create EldenGymEnv
   ↓
2. Create EldenClient
   ↓
3. EldenClient.load_config_from_file()
   ↓
4. Parse TOML config → extract patterns
   ↓
5. Send config to Siphon server
   ↓
6. Initialize memory subsystem
   ├─ Find process by name
   ├─ Scan for patterns
   └─ Build pointer chains
   ↓
7. Initialize input subsystem
   └─ Attach to game window
   ↓
8. Initialize capture subsystem
   └─ Set up DXGI capture
   ↓
9. Ready for training!
```

### Training Step Flow

```
1. Agent selects action
   ↓
2. EldenGymEnv.step(action)
   ↓
3. Convert action to inputs
   ↓
4. EldenClient.input_key_tap(keys, time)
   ↓
5. SiphonClient (gRPC call)
   ↓
6. Siphon Server → inject input
   ↓
7. Game processes input (frame_skip frames)
   ↓
8. Read game state
   ├─ EldenClient.player_hp
   ├─ EldenClient.target_hp
   └─ EldenClient.capture_frame()
   ↓
9. Compute reward
   ↓
10. Return (obs, reward, terminated, truncated, info)
```

## Design Decisions

### Why gRPC?

**Pros:**
- High performance (Protocol Buffers)
- Language agnostic (C++ server, Python client)
- Streaming support for frames
- Built-in error handling

**Alternatives considered:**
- REST API (too slow for real-time)
- Shared memory (platform-specific)
- Sockets (manual protocol design)

### Why C++ Server?

**Pros:**
- Low-level Windows API access
- Direct memory manipulation
- High performance capture (DXGI)
- Minimal overhead

**Alternatives:**
- Pure Python (too slow, no low-level access)
- Rust (considered, but C++ has more Windows examples)

### Why TOML Config?

**Pros:**
- Human-readable
- Easy to edit
- Built-in Python support (tomllib/tomli)
- Structured data

**Alternatives:**
- JSON (less readable)
- YAML (complex, security issues)
- Python files (not data-focused)

### Why Gymnasium?

**Pros:**
- Standard RL interface
- Wide ecosystem support
- Well-documented
- Active development

**Alternatives:**
- Custom API (wheel reinvention)
- OpenAI Gym (deprecated)

## Performance Considerations

### Frame Capture

**Challenge:** Capturing frames is expensive

**Solutions:**
- DXGI GPU capture (faster than GDI)
- Frame skipping (default: 4 frames)
- Optional frame capture (dict observation mode)
- Efficient frame encoding

### Memory Reading

**Challenge:** Frequent memory reads can be slow

**Solutions:**
- Batch reads when possible
- Pointer caching
- Efficient gRPC serialization

### Input Injection

**Challenge:** Input timing is critical

**Solutions:**
- Direct input injection (bypasses window focus)
- Precise timing control (milliseconds)
- Frame-synchronized actions

## Extensibility

### Adding New Games

1. Create game-specific client (inherit from `SiphonClient`)
2. Create TOML config with memory patterns
3. Define scenarios
4. Create Gymnasium environment

### Adding New Scenarios

```python
# In elden_client.py
self.scenarios = {
    "margit": {
        "boss_name": "Margit",
        "fog_wall_location": (x, y, z),
    },
    "godrick": {  # New scenario
        "boss_name": "Godrick",
        "fog_wall_location": (x, y, z),
    }
}
```

### Custom Reward Functions

```python
def custom_reward(obs, info, terminated, truncated):
    # Your logic
    return reward

env = gym.make("EldenGym-v0", reward_function=custom_reward)
```

### Custom Wrappers

Use Gymnasium wrappers for preprocessing, augmentation, etc.

## Security Considerations

⚠️ **Warning:** This tool manipulates game memory and injects inputs.

**Safety measures:**
- Local-only by default (localhost:50051)
- No remote code execution
- Read-only mode possible
- Sandboxed game instance recommended

**Use responsibly:**
- Only for single-player
- Respect game ToS
- Educational/research purposes

## Future Improvements

- [ ] Multi-agent support
- [ ] Replay buffer integration
- [ ] Better state serialization
- [ ] Cloud training support
- [ ] Web dashboard for monitoring
- [ ] More boss scenarios
- [ ] PvP scenarios
- [ ] Speedrun scenarios

## Resources

- [Gymnasium Docs](https://gymnasium.farama.org/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
- [Memory Scanning Basics](https://guidedhacking.com/)
