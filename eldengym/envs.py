"""
Register all default EldenGym environments.

This module registers pre-configured environments for common boss fights.
"""

from pathlib import Path
from .env import EldenGymEnv
from .registry import register

# Get the package directory for resolving file paths
PACKAGE_DIR = Path(__file__).parent
FILES_DIR = PACKAGE_DIR / "files"

# Register Margit environments
register(
    id="Margit-v0",
    entry_point=EldenGymEnv,
    kwargs={
        "scenario_name": "Margit-v0",
        "keybinds_filepath": str(FILES_DIR / "Margit-v0" / "keybinds_v2.json"),
        "siphon_config_filepath": str(
            FILES_DIR / "Margit-v0" / "er_siphon_config.toml"
        ),
    },
)
