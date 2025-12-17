# Contributing to EldenGym

Thank you for your interest in contributing to EldenGym!

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/dhmnr/eldengym.git
cd eldengym
```

### 2. Install Development Dependencies

```bash
# Install with dev dependencies using uv
uv sync --group dev --group docs

# Or with pip
pip install -e ".[dev,docs]"
```

### 3. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

## Development Workflow

### Code Style

We use `ruff` for linting and formatting:

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

### Testing

```bash
# Run tests (when implemented)
uv run pytest

# Run specific test
uv run pytest tests/test_env.py

# With coverage
uv run pytest --cov=eldengym
```

### Documentation

Build and serve documentation locally:

```bash
# Install docs dependencies
uv sync --group docs

# Serve docs locally
uv run mkdocs serve

# Build docs
uv run mkdocs build
```

Visit `http://localhost:8000` to see the documentation.

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-bugfix
```

### 2. Make Your Changes

- Write clear, documented code
- Add docstrings to new functions/classes
- Update documentation if needed
- Add tests for new features

### 3. Commit

We use semantic commit messages:

```bash
# Feature
git commit -m "feat(env): add new observation type"

# Bug fix
git commit -m "fix(client): resolve connection timeout"

# Documentation
git commit -m "docs: update installation guide"

# Breaking change
git commit -m "feat(api)!: redesign action space

BREAKING CHANGE: Action space now uses continuous values"
```

### 4. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub.

## Contribution Areas

### High Priority

- 🧪 **Tests** - Add unit and integration tests
- 📚 **Documentation** - Improve guides and examples
- 🎮 **Scenarios** - Add new boss fight scenarios
- 🎁 **Wrappers** - Create useful environment wrappers

### Features

- **Reward Functions** - New reward function implementations
- **Observation Processing** - Better frame preprocessing
- **Action Spaces** - Alternative action representations
- **Memory Patterns** - Support for different game versions

### Bug Fixes

- Performance improvements
- Memory leaks
- Connection issues
- Game state synchronization

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Docstrings: Google style

```python
def my_function(arg1: int, arg2: str) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong
    """
    return True
```

### Docstring Examples

```python
class MyClass:
    """Brief description.

    Detailed description of the class purpose and usage.

    Attributes:
        attr1: Description
        attr2: Description

    Example:
        >>> obj = MyClass()
        >>> obj.method()
        'result'
    """

    def method(self) -> str:
        """Method description."""
        return "result"
```

## Project Structure

```
eldengym/
  ├── eldengym/
  │   ├── __init__.py
  │   ├── env.py          # Main environment
  │   ├── envs.py         # Environment registration
  │   ├── rewards.py      # Reward functions
  │   ├── wrappers.py     # Gymnasium wrappers
  │   ├── utils.py        # Utility functions
  │   ├── registry.py     # Scenario registry
  │   ├── client/
  │   │   └── elden_client.py    # Game-specific client (uses pysiphon)
  │   └── files/
  │       └── configs/    # Game configurations
  ├── examples/           # Example notebooks
  ├── docs/              # Documentation
  ├── tests/             # Test suite
  └── pyproject.toml     # Project config
```

## Getting Help

- **Issues**: Open an issue for bugs or features
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: (link if available)

## Code of Conduct

Be respectful and constructive. We're all here to learn and build something cool!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
