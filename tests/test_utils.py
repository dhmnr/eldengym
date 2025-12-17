"""Test utility functions."""

from pathlib import Path
from eldengym.utils import resolve_file_path


def test_resolve_file_path_absolute():
    """Test resolving absolute paths."""
    abs_path = Path("/absolute/path/file.txt")
    resolved = resolve_file_path(abs_path)
    assert resolved == abs_path


def test_resolve_file_path_relative():
    """Test resolving relative paths."""
    rel_path = "files/config.toml"
    resolved = resolve_file_path(rel_path, relative_to_package=True)
    assert resolved.is_absolute()
    assert "eldengym" in str(resolved)


def test_resolve_file_path_no_package():
    """Test resolving without package reference."""
    rel_path = "test.txt"
    resolved = resolve_file_path(rel_path, relative_to_package=False)
    assert resolved.is_absolute()
