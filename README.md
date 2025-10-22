# EldenGym 🎮

[![PyPI version](https://badge.fury.io/py/eldengym.svg)](https://pypi.org/project/eldengym/)
[![Tests](https://github.com/dhmnr/eldengym/actions/workflows/test.yml/badge.svg)](https://github.com/dhmnr/eldengym/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-cloudflare%20pages-orange)](https://eldengym.dhmnr.sh/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Gymnasium-compatible reinforcement learning environment for Elden Ring.

## 🚀 Quick Start

```python
import gymnasium as gym
import eldengym

# Create environment
env = gym.make("EldenGym-v0", scenario_name="margit")

# Standard RL loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## ✨ Features

- 🎯 **Gymnasium API** - Standard RL interface
- 🎮 **Multiple Action Spaces** - Discrete, multi-binary, or continuous
- 📊 **Flexible Observations** - RGB frames, game state, or both
- ⚡ **High Performance** - gRPC-based C++ backend
- 🔧 **Customizable** - Easy reward functions and wrappers
- 🏆 **Boss Scenarios** - Pre-configured boss fights

## 📚 Documentation

**[Read the full documentation →](https://eldengym.dhmnr.sh/)**

- [Installation Guide](https://eldengym.dhmnr.sh/getting-started/installation/)
- [Quick Start Tutorial](https://eldengym.dhmnr.sh/getting-started/quickstart/)
- [API Reference](https://eldengym.dhmnr.sh/api/env/)
- [Examples](https://eldengym.dhmnr.sh/examples/random_policy/)

## 🔧 Installation

### From PyPI (Stable Release)

```bash
pip install eldengym
```

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/dhmnr/eldengym.git
cd eldengym

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## 🎮 Usage Examples

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("EldenGym-v0", scenario_name="margit")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### Custom Reward Function

```python
def custom_reward(obs, info, terminated, truncated):
    reward = 0.0
    if 'target_hp_delta' in info:
        reward += info['target_hp_delta'] * 10.0
    return reward

env = gym.make("EldenGym-v0", reward_function=custom_reward)
```

## 📖 Examples

Check out the [examples](examples/) directory:

- [`random_policy.ipynb`](examples/random_policy.ipynb) - Random agent baseline
- [`llm_agent.ipynb`](examples/llm_agent.ipynb) - LLM-based agent

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --group dev --group test --group docs

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Serve documentation locally
uv run mkdocs serve
```

## 📋 Requirements

- **Elden Ring** (PC version)
- **Siphon Server** (C++ gRPC backend)
- **Python 3.10+**
- **Windows** (for game interaction)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://eldengym.dhmnr.sh/development/contributing/).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- Documentation powered by [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/dhmnr/eldengym/issues)
- Discussions: [GitHub Discussions](https://github.com/dhmnr/eldengym/discussions)

---

⭐ Star this repo if you find it useful!
