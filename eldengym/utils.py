"""
Utility functions for EldenGym.
"""

from pathlib import Path


def resolve_file_path(filepath, relative_to_package=True):
    """
    Resolve a file path relative to the package or as absolute path.

    Args:
        filepath: str or Path, file path to resolve
        relative_to_package: bool, if True resolves relative to eldengym package

    Returns:
        Path: Resolved absolute path

    Example:
        >>> path = resolve_file_path("files/config.toml")
        >>> print(path)
    """
    path = Path(filepath)

    if path.is_absolute():
        return path

    if relative_to_package:
        # Resolve relative to eldengym package directory
        package_dir = Path(__file__).parent
        return (package_dir / path).resolve()

    return path.resolve()
