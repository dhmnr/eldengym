"""Tests for utility functions."""

import pytest
from pathlib import Path
from eldengym.utils import parse_config_file


def test_parse_config_file_valid():
    """Test parsing a valid config file."""
    # Using the actual ER config file
    config_path = (
        Path(__file__).parent.parent
        / "eldengym"
        / "files"
        / "configs"
        / "ER_1_16_1.toml"
    )

    if config_path.exists():
        process_name, window_name, attributes = parse_config_file(config_path)

        assert process_name == "eldenring.exe"
        assert window_name == "ELDEN RING"
        assert len(attributes) > 0
        assert all("name" in attr for attr in attributes)
        assert all("pattern" in attr for attr in attributes)
        assert all("offsets" in attr for attr in attributes)
        assert all("type" in attr for attr in attributes)


def test_parse_config_file_not_found():
    """Test parsing a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        parse_config_file("nonexistent.toml")


def test_parse_config_file_invalid_structure(tmp_path):
    """Test parsing a config file with invalid structure."""
    # Create a TOML file without required sections
    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text("[some_section]\nkey = 'value'")

    with pytest.raises(ValueError, match="Missing.*section"):
        parse_config_file(invalid_config)
