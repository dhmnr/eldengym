"""
Utility functions for EldenGym.
"""

import sys
from pathlib import Path

# Handle TOML parsing for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("For Python < 3.11, please install tomli: pip install tomli")


def parse_config_file(config_filepath):
    """
    Parse a TOML config file and extract process configuration and attributes.

    This function mirrors the ParseConfigFile function from the C++ client,
    reading process information and memory attributes from a TOML configuration file.

    Args:
        config_filepath: str or Path, path to the TOML configuration file

    Returns:
        tuple: (process_name, process_window_name, attributes)
            - process_name: str, name of the process (e.g., "eldenring.exe")
            - process_window_name: str, window name of the process (e.g., "ELDEN RING")
            - attributes: list of dicts, each containing:
                - name: str, attribute name
                - pattern: str, memory pattern to search for
                - offsets: list of int, memory offsets
                - type: str, data type (e.g., "int", "float", "array")
                - length: int, length for array types (default: 0)
                - method: str, method for reading (default: "")

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is malformed or missing required fields

    Example:
        >>> process_name, window_name, attrs = parse_config_file("config.toml")
        >>> print(f"Process: {process_name}, Window: {window_name}")
        >>> print(f"Attributes: {len(attrs)}")
    """
    config_path = Path(config_filepath)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_filepath}")

    # Read and parse TOML file
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Extract process info
    if "process_info" not in config:
        raise ValueError("Missing [process_info] section in config file")

    process_info = config["process_info"]
    process_name = process_info.get("name", "")
    process_window_name = process_info.get("window_name", "")

    if not process_name:
        raise ValueError("Missing 'name' field in [process_info] section")

    # Extract attributes
    if "attributes" not in config:
        raise ValueError("Missing [attributes] section in config file")

    attributes = []
    attributes_section = config["attributes"]

    for attr_name, attr_data in attributes_section.items():
        if not isinstance(attr_data, dict):
            continue

        attribute = {
            "name": attr_name,
            "pattern": attr_data.get("pattern", ""),
            "offsets": attr_data.get("offsets", []),
            "type": attr_data.get("type", ""),
            "length": attr_data.get("length", 0),
            "method": attr_data.get("method", ""),
        }

        attributes.append(attribute)

    return process_name, process_window_name, attributes
