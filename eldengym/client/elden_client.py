from pysiphon import SiphonClient
import numpy as np
from pathlib import Path
from time import sleep


class EldenClient(SiphonClient):
    """
    Client for the Elden Ring game.
    """

    def __init__(self, host="localhost:50051", **kwargs):
        super().__init__(host, **kwargs)
        self.scenarios = {
            "Margit-v0": {
                "boss_name": "Margit",
                "fog_wall_location": (
                    19.958229064941406,
                    -11.990748405456543,
                    -7.051832675933838,
                ),
            }
        }

    ## =========== Initialization methods ===========

    def _resolve_config_path(self, config_filepath):
        """
        Resolve config file path relative to package root.

        Args:
            config_filepath: str or Path, can be:
                - Absolute path: used as-is
                - Relative path: resolved relative to package root (eldengym/)
                - Filename only: looked up in eldengym/files/configs/

        Returns:
            Path: Resolved absolute path to config file
        """
        config_path = Path(config_filepath)

        # If absolute path, use it directly
        if config_path.is_absolute():
            return config_path

        # Get package root (eldengym/)
        package_root = Path(__file__).parent.parent

        # If it's just a filename (no directory parts), look in configs directory
        if len(config_path.parts) == 1:
            config_path = package_root / "files" / "configs" / config_path
        else:
            # Relative path - resolve from package root
            config_path = package_root / config_path

        return config_path.resolve()

    def load_config_from_file(self, config_filepath, wait_time=2):
        """
        Complete initialization sequence: load config, initialize memory, input, and capture.

        This is a convenience method that performs all initialization steps at once,
        mirroring the 'init' command from the C++ client.

        Args:
            config_filepath: str or Path, path to TOML config file. Can be:
                - Absolute path: /full/path/to/config.toml
                - Relative to package: files/configs/ER_1_16_1.toml
                - Filename only: ER_1_16_1.toml (searches in eldengym/files/configs/)
            wait_time: int, seconds to wait after loading config before initializing subsystems

        Returns:
            dict with keys 'config', 'memory', 'input', 'capture' containing the response dictionaries

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is malformed
            RuntimeError: If any initialization step fails

        Example:
            >>> client = EldenClient()
            >>> # All these work from any directory:
            >>> results = client.load_config_from_file("ER_1_16_1.toml")
            >>> results = client.load_config_from_file("files/configs/ER_1_16_1.toml")
            >>> results = client.load_config_from_file("/absolute/path/to/config.toml")
        """
        import time

        results = {}

        # Resolve config path
        resolved_path = self._resolve_config_path(config_filepath)

        # pysiphon's set_process_config reads the TOML file directly
        print(f"Loading config from: {resolved_path}")
        print("Sending configuration to server...")
        config_response = self.set_process_config(str(resolved_path))
        results["config"] = config_response

        if not config_response.get("success", False):
            raise RuntimeError(
                f"Failed to set process config: {config_response.get('message', 'Unknown error')}"
            )

        print(f"Server response: {config_response.get('message', 'Success')}")

        # Wait for process to be ready
        if wait_time > 0:
            print(f"Waiting {wait_time} seconds for process to be ready...")
            time.sleep(wait_time)

        # Initialize memory
        print("Initializing memory subsystem...")
        memory_response = self.initialize_memory()
        results["memory"] = memory_response

        if not memory_response.get("success", False):
            raise RuntimeError(
                f"Failed to initialize memory: {memory_response.get('message', 'Unknown error')}"
            )

        print(f"Server response: {memory_response.get('message', 'Success')}")
        if memory_response.get("process_id", 0) > 0:
            print(f"Process ID: {memory_response['process_id']}")

        # Initialize input
        print("Initializing input subsystem...")
        input_response = self.initialize_input()
        results["input"] = input_response

        if not input_response.get("success", False):
            raise RuntimeError(
                f"Failed to initialize input: {input_response.get('message', 'Unknown error')}"
            )

        print(f"Server response: {input_response.get('message', 'Success')}")

        # Initialize capture
        print("Initializing capture subsystem...")
        capture_response = self.initialize_capture()
        results["capture"] = capture_response

        if not capture_response.get("success", False):
            raise RuntimeError(
                f"Failed to initialize capture: {capture_response.get('message', 'Unknown error')}"
            )

        print(f"Server response: {capture_response.get('message', 'Success')}")
        window_width = capture_response.get("window_width", 0)
        window_height = capture_response.get("window_height", 0)
        if window_width > 0 and window_height > 0:
            print(f"Window size: {window_width}x{window_height}")

        print("\n=== Initialization Complete! ===")
        print("All subsystems initialized successfully.")

        return results

    def launch_game(self):
        """
        Launch the game.
        """
        launch_response = self.execute_command(
            "start_protected_game.exe",
            args=None,
            working_directory=r"C:\Program Files (x86)\Steam\steamapps\common\ELDEN RING\Game",
        )
        if not launch_response.get("success", False):
            raise RuntimeError(
                f"Failed to launch game: {launch_response.get('message', 'Unknown error')}"
            )

        print(f"Server response: {launch_response.get('message', 'Success')}")
        if launch_response.get("process_id", 0) > 0:
            print(f"Process ID: {launch_response['process_id']}")

        return launch_response

    ## =========== Player methods ===========
    @property
    def player_hp(self):
        """
        Get the health of the player.

        Returns:
            int: Current HP value
        """
        response = self.get_attribute("HeroHp")
        return response.get("value", 0) if isinstance(response, dict) else response

    @property
    def player_max_hp(self):
        """
        Get the maximum health of the player.

        Returns:
            int: Maximum HP value
        """
        response = self.get_attribute("HeroMaxHp")
        return response.get("value", 0) if isinstance(response, dict) else response

    def set_player_hp(self, hp):
        """
        Set the health of the player.

        Args:
            hp (int): HP value to set
        """
        self.set_attribute("HeroHp", hp, "int")

    @property
    def local_player_coords(self):
        """
        Get the local coordinates of the player.

        Returns:
            tuple: (x, y, z) local coordinates
        """
        local_x = self.get_attribute("HeroLocalPosX")
        local_y = self.get_attribute("HeroLocalPosY")
        local_z = self.get_attribute("HeroLocalPosZ")

        # Extract values from dict responses
        local_x = local_x.get("value", 0.0) if isinstance(local_x, dict) else local_x
        local_y = local_y.get("value", 0.0) if isinstance(local_y, dict) else local_y
        local_z = local_z.get("value", 0.0) if isinstance(local_z, dict) else local_z

        return local_x, local_y, local_z

    @property
    def global_player_coords(self):
        """
        Get the global coordinates of the player.

        Returns:
            tuple: (x, y, z) global coordinates
        """
        global_x = self.get_attribute("HeroGlobalPosX")
        global_y = self.get_attribute("HeroGlobalPosY")
        global_z = self.get_attribute("HeroGlobalPosZ")

        # Extract values from dict responses
        global_x = (
            global_x.get("value", 0.0) if isinstance(global_x, dict) else global_x
        )
        global_y = (
            global_y.get("value", 0.0) if isinstance(global_y, dict) else global_y
        )
        global_z = (
            global_z.get("value", 0.0) if isinstance(global_z, dict) else global_z
        )

        return global_x, global_y, global_z

    @property
    def player_animation_id(self):
        """
        Get the animation id of the player.

        Returns:
            int: Animation ID
        """
        response = self.get_attribute("HeroAnimId")
        return response.get("value", 0) if isinstance(response, dict) else response

    ## =========== Target methods ===========
    @property
    def target_hp(self):
        """
        Get the health of the target.

        Returns:
            int: Current target HP value
        """
        response = self.get_attribute("NpcHp")
        return response.get("value", 0) if isinstance(response, dict) else response

    @property
    def target_max_hp(self):
        """
        Get the maximum health of the target.

        Returns:
            int: Maximum target HP value
        """
        response = self.get_attribute("NpcMaxHp")
        return response.get("value", 0) if isinstance(response, dict) else response

    def set_target_hp(self, hp):
        """
        Set the health of the target.

        Args:
            hp (int): HP value to set
        """
        self.set_attribute("NpcHp", hp, "int")

    @property
    def local_target_coords(self):
        """
        Get the local coordinates of the target.

        Returns:
            tuple: (x, y, z) local coordinates
        """
        local_x = self.get_attribute("NpcLocalPosX")
        local_y = self.get_attribute("NpcLocalPosY")
        local_z = self.get_attribute("NpcLocalPosZ")

        # Extract values from dict responses
        local_x = local_x.get("value", 0.0) if isinstance(local_x, dict) else local_x
        local_y = local_y.get("value", 0.0) if isinstance(local_y, dict) else local_y
        local_z = local_z.get("value", 0.0) if isinstance(local_z, dict) else local_z

        return local_x, local_y, local_z

    @property
    def global_target_coords(self):
        """
        Get the global coordinates of the target.

        Returns:
            tuple: (x, y, z) global coordinates
        """
        global_x = self.get_attribute("NpcGlobalPosX")
        global_y = self.get_attribute("NpcGlobalPosY")
        global_z = self.get_attribute("NpcGlobalPosZ")

        # Extract values from dict responses
        global_x = (
            global_x.get("value", 0.0) if isinstance(global_x, dict) else global_x
        )
        global_y = (
            global_y.get("value", 0.0) if isinstance(global_y, dict) else global_y
        )
        global_z = (
            global_z.get("value", 0.0) if isinstance(global_z, dict) else global_z
        )

        return global_x, global_y, global_z

    @property
    def target_animation_id(self):
        """
        Get the animation id of the target.

        Returns:
            int: Animation ID
        """
        response = self.get_attribute("NpcAnimId")
        return response.get("value", 0) if isinstance(response, dict) else response

    ## =========== Helper methods ===========
    @property
    def target_player_distance(self):
        """
        Get the distance between the player and the target.
        """
        player_x, player_y, player_z = self.local_player_coords
        target_x, target_y, target_z = self.global_target_coords
        return np.linalg.norm(
            [player_x - target_x, player_y - target_y, player_z - target_z]
        )

    def teleport(self, x, y, z):
        """
        Teleport the player to the given global coordinates.

        Takes target global coords, computes delta, writes to local coords.

        Args:
            x (float): Target global X coordinate
            y (float): Target global Y coordinate
            z (float): Target global Z coordinate
        """
        local_x, local_y, local_z = self.local_player_coords
        global_x, global_y, global_z = self.global_player_coords
        self.set_attribute("HeroLocalPosX", local_x + (x - global_x), "float")
        self.set_attribute("HeroLocalPosY", local_y + (y - global_y), "float")
        self.set_attribute("HeroLocalPosZ", local_z + (z - global_z), "float")

    def teleport_to(self, x, y, z):
        """Alias for teleport(). Teleport player to global coordinates."""
        self.teleport(x, y, z)

    def set_game_speed(self, speed):
        """
        Set the game speed.

        Args:
            speed (float): Game speed multiplier (e.g., 0.5 for half speed, 2.0 for double speed)
        """
        self.set_attribute("gameSpeedFlag", True, "bool")
        self.set_attribute("gameSpeedVal", speed, "float")

    def start_scenario(self, scenario_name="Margit"):
        """
        Start the scenario with the given scenario name.
        """
        # FIXME: This is a hack to start boss fight. Need to check fogwall state. or use another method.
        x, y, z = self.scenarios[scenario_name]["fog_wall_location"]
        self.teleport(x, y, z)
        self.move_mouse(1000, 0, 1)
        sleep(2)
        self.input_key_tap(["W", "E"], 200, 200)
        sleep(2)
        self.input_key_tap(["B"], 200)

    ## =========== Save File Management ===========
    def copy_save_file(self, save_file_name, save_file_dir=None, timeout_seconds=10):
        """
        Copy a backup save file to become the active save file.

        """
        import os

        # Use default Elden Ring save directory if not provided
        if save_file_dir is None:
            # Default path - user will need to replace with their Steam ID
            save_file_dir = os.path.join(
                os.getenv("APPDATA"),
                "EldenRing",
                # Note: Need to append the actual Steam ID subdirectory
            )
            print(f"[Warning] Using default save directory: {save_file_dir}")
            print(
                "[Warning] You may need to specify the full path including your Steam ID"
            )

        source_path = os.path.join(save_file_dir, save_file_name)
        dest_path = os.path.join(save_file_dir, "ER0000.sl2")

        print("Copying save file:")
        print(f"  Source: {source_path}")
        print(f"  Dest:   {dest_path}")

        # Use PowerShell to copy the file
        result = self.execute_command(
            "powershell",
            args=[
                "-Command",
                f'Copy-Item -Path "{source_path}" -Destination "{dest_path}" -Force',
            ],
            timeout_seconds=timeout_seconds,
            capture_output=True,
        )

        if not result.get("success", False):
            raise RuntimeError(
                f"Failed to copy save file: {result.get('message', 'Unknown error')}"
            )

        print("✓ Save file copied successfully")
        sleep(2.0)  # Wait for filesystem to sync

        return result

    ## =========== Menu methods ===========
    def enter_menu(self):
        """
        Enter the menu.
        """
        self.input_key_tap(["ENTER"], 200, 0)
        sleep(1)
        self.input_key_tap(["ENTER"], 200, 0)
        sleep(1)
        self.input_key_tap(["ENTER"], 200, 0)
        sleep(10)

    def quit_to_title(self):
        """
        Quit the game to the title screen.
        """
        self.input_key_tap(["ESC"])
        sleep(0.3)

        # Navigate menu (UP_ARROW, E)
        self.input_key_tap(["UP_ARROW"])
        sleep(0.3)
        self.input_key_tap(["E"])
        sleep(0.3)

        # Confirm quit (Z, E, LEFT_ARROW, E)
        self.input_key_tap(["Z"])
        sleep(0.3)
        self.input_key_tap(["E"])
        sleep(0.3)
        self.input_key_tap(["LEFT_ARROW"])
        sleep(0.3)
        self.input_key_tap(["E"])
        sleep(12.0)
        sleep(5.0)
