# Tests

Test suite for EldenGym.

## Running Tests

### Install test dependencies

```bash
uv sync --group test
```

### Run all tests

```bash
uv run pytest
```

### Run with coverage

```bash
uv run pytest --cov=eldengym --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Run specific tests

```bash
# Run a specific test file
uv run pytest tests/test_utils.py

# Run a specific test
uv run pytest tests/test_utils.py::test_parse_config_file_valid

# Run tests matching a pattern
uv run pytest -k "parse"
```

### Run tests with markers

```bash
# Run only unit tests
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py
├── test_import.py        # Import tests
├── test_utils.py         # Utility function tests
└── README.md            # This file
```

## Writing Tests

### Test naming convention

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Example test

```python
import pytest
from eldengym.utils import parse_config_file

def test_parse_config_file_valid():
    """Test parsing a valid config file."""
    process_name, window_name, attributes = parse_config_file("config.toml")
    assert process_name == "eldenring.exe"
    assert len(attributes) > 0
```

### Using markers

```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_with_server():
    """This test requires a running server."""
    pass
```

## CI/CD

Tests run automatically on:
- Every push to `main`
- Every pull request

Test matrix:
- **OS**: Ubuntu, Windows, macOS
- **Python**: 3.10, 3.11, 3.12

## Coverage

Coverage reports are generated and uploaded to Codecov on every CI run.

Target: 80%+ coverage
