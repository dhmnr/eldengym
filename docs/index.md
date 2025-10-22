# EldenGym

Welcome to **EldenGym** - a Gymnasium environment for training reinforcement learning agents in Elden Ring!

## Overview

EldenGym provides a complete RL environment interface for Elden Ring, allowing you to:

- Train agents on boss fights and game scenarios
- Access game state through memory reading
- Control the game through automated inputs
- Capture and process game frames in real-time
- Customize reward functions and observation spaces

## Features

✨ **Full Gymnasium API** - Standard RL environment interface
🎮 **Game Control** - Keyboard, mouse, and game state manipulation
📊 **Flexible Observations** - RGB frames, game state, or both
🎯 **Custom Rewards** - Define your own reward functions
⚡ **High Performance** - gRPC-based communication with C++ backend
🔧 **Easy Configuration** - TOML-based game configuration

## Quick Example

```python
import gymnasium as gym
import eldengym

# Create the environment
env = gym.make("EldenGym-v0", scenario_name="margit")

# Standard RL loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Architecture

```
┌─────────────────────────────────────┐
│        Your RL Agent                │
│  (stable-baselines3, custom, etc)   │
└──────────────┬──────────────────────┘
               │ Gymnasium API
               ▼
┌─────────────────────────────────────┐
│         EldenGym (Python)           │
│  - Environment wrapper              │
│  - Reward functions                 │
│  - Observation processing           │
└──────────────┬──────────────────────┘
               │ gRPC
               ▼
┌─────────────────────────────────────┐
│      Siphon Server (C++)            │
│  - Memory reading/writing           │
│  - Input injection                  │
│  - Screen capture                   │
└──────────────┬──────────────────────┘
               │
               ▼
         Elden Ring Game
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Get started with EldenGym
- [Quick Start Tutorial](getting-started/quickstart.md) - Your first RL agent
- [API Reference](api/env.md) - Detailed API documentation
- [Examples](examples/random_policy.ipynb) - See EldenGym in action

## Citation

If you use EldenGym in your research, please cite:

```bibtex
@software{eldengym2025,
  title = {EldenGym: A Gymnasium Environment for Elden Ring},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/dhmnr/eldengym}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
