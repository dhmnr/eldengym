from .siphon_client import SiphonClient
from ..utils import parse_config_file
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
            "margit": {
                "boss_name": "Margit",
                "fog_wall_location": (
                    19.958229064941406,
                    -11.990748405456543,
                    -7.051832675933838,
                ),
            }
        }

    ## =========== Initialization methods ===========
    # def load_config_from_file(self, config_filepath):
    #     """
    #     Load process configuration from a TOML file and send to server.

    #     This is a convenience method that parses the config file and calls
    #     set_process_config with the extracted data.

    #     Args:
    #         config_filepath: str or Path, path to TOML config file

    #     Returns:
    #         SetProcessConfigResponse from server

    #     Raises:
    #         FileNotFoundError: If config file doesn't exist
    #         ValueError: If config file is malformed

    #     Example:
    #         >>> client = EldenClient()
    #         >>> response = client.load_config_from_file("config.toml")
    #         >>> if response.success:
    #         ...     print("Config loaded successfully!")
    #     """
    #     process_name, process_window_name, attributes = parse_config_file(config_filepath)
    #     return self.set_process_config(process_name, process_window_name, attributes)

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
            dict with keys 'config', 'memory', 'input', 'capture' containing the responses

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

        # Parse config file
        print(f"Loading config from: {resolved_path}")
        process_name, process_window_name, attributes = parse_config_file(resolved_path)

        print(
            f"Config loaded - Process: {process_name}, Window: {process_window_name}, "
            f"Attributes: {len(attributes)}"
        )

        # Send config to server
        print("Sending configuration to server...")
        config_response = self.set_process_config(
            process_name, process_window_name, attributes
        )
        results["config"] = config_response

        if not config_response.success:
            raise RuntimeError(
                f"Failed to set process config: {config_response.message}"
            )

        print(f"Server response: {config_response.message}")

        # Wait for process to be ready
        if wait_time > 0:
            print(f"Waiting {wait_time} seconds for process to be ready...")
            time.sleep(wait_time)

        # Initialize memory
        print("Initializing memory subsystem...")
        memory_response = self.initialize_memory()
        results["memory"] = memory_response

        if not memory_response.success:
            raise RuntimeError(
                f"Failed to initialize memory: {memory_response.message}"
            )

        print(f"Server response: {memory_response.message}")
        if hasattr(memory_response, "process_id") and memory_response.process_id > 0:
            print(f"Process ID: {memory_response.process_id}")

        # Initialize input
        print("Initializing input subsystem...")
        input_response = self.initialize_input()
        results["input"] = input_response

        if not input_response.success:
            raise RuntimeError(f"Failed to initialize input: {input_response.message}")

        print(f"Server response: {input_response.message}")

        # Initialize capture
        print("Initializing capture subsystem...")
        capture_response = self.initialize_capture()
        results["capture"] = capture_response

        if not capture_response.success:
            raise RuntimeError(
                f"Failed to initialize capture: {capture_response.message}"
            )

        print(f"Server response: {capture_response.message}")
        if hasattr(capture_response, "window_width") and hasattr(
            capture_response, "window_height"
        ):
            if capture_response.window_width > 0 and capture_response.window_height > 0:
                print(
                    f"Window size: {capture_response.window_width}x{capture_response.window_height}"
                )

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
            working_directory="C:\Program Files (x86)\Steam\steamapps\common\ELDEN RING\Game",
        )
        if not launch_response.success:
            raise RuntimeError(f"Failed to launch game: {launch_response.message}")

        print(f"Server response: {launch_response.message}")
        if hasattr(launch_response, "process_id") and launch_response.process_id > 0:
            print(f"Process ID: {launch_response.process_id}")

        return launch_response

    def bypass_menu(self):
        """
        Bypass the menu.
        """
        self.send_key(["ENTER"], 200, 0)
        sleep(1)
        self.send_key(["ENTER"], 200, 0)
        sleep(1)
        self.send_key(["ENTER"], 200, 0)

    ## =========== Player methods ===========
    @property
    def player_hp(self):
        """
        Get the health of the player.
        """
        return self.get_attribute("HeroHp")

    @property
    def player_max_hp(self):
        """
        Get the maximum health of the player.
        """
        return self.get_attribute("HeroMaxHp")

    def set_player_hp(self, hp):
        """
        Set the health of the player.
        """
        self.set_attribute("HeroHp", hp)

    @property
    def local_player_coords(self):
        """
        Get the location of the player.
        """
        local_x = self.get_attribute("HeroLocalPosX")
        local_y = self.get_attribute("HeroLocalPosY")
        local_z = self.get_attribute("HeroLocalPosZ")
        return local_x, local_y, local_z

    @property
    def global_player_coords(self):
        """
        Get the location of the player.
        """
        global_x = self.get_attribute("HeroGlobalPosX")
        global_y = self.get_attribute("HeroGlobalPosY")
        global_z = self.get_attribute("HeroGlobalPosZ")
        return global_x, global_y, global_z

    @property
    def player_animation_id(self):
        """
        Get the animation id of the player.
        """
        return self.get_attribute("HeroAnimId")

    ## =========== Target methods ===========
    @property
    def target_hp(self):
        """
        Get the health of the target.
        """
        return self.get_attribute("NpcHp")

    @property
    def target_max_hp(self):
        """
        Get the maximum health of the target.
        """
        return self.get_attribute("NpcMaxHp")

    def set_target_hp(self, hp):
        """
        Set the health of the target.
        """
        self.set_attribute("NpcHp", hp)

    @property
    def local_target_coords(self):
        """
        Get the location of the target.
        """
        local_x = self.get_attribute("NpcLocalPosX")
        local_y = self.get_attribute("NpcLocalPosY")
        local_z = self.get_attribute("NpcLocalPosZ")
        return local_x, local_y, local_z

    @property
    def global_target_coords(self):
        """
        Get the location of the target.
        """
        global_x = self.get_attribute("NpcGlobalPosX")
        global_y = self.get_attribute("NpcGlobalPosY")
        global_z = self.get_attribute("NpcGlobalPosZ")
        return global_x, global_y, global_z

    @property
    def target_animation_id(self):
        """
        Get the animation id of the target.
        """
        return self.get_attribute("NpcAnimId")

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
        Teleport the player to the given coordinates.
        """
        # FIXME: Close range teleport, need to check MapId for long range teleport.
        local_x, local_y, local_z = self.local_player_coords
        global_x, global_y, global_z = self.global_player_coords
        self.set_attribute("HeroLocalPosX", local_x + (x - global_x))
        self.set_attribute("HeroLocalPosY", local_y + (y - global_y))
        self.set_attribute("HeroLocalPosZ", local_z + (z - global_z))

    def set_game_speed(self, speed):
        """
        Set the game speed.
        """
        self.set_attribute("gameSpeedFlag", True)
        self.set_attribute("gameSpeedVal", speed)

    def reset_game(self):
        """
        Reset the game by setting the player's hp to 0.
        """
        self.set_player_hp(0)
        sleep(
            20
        )  # FIXME: This is a hack to wait for the game to reset, doesn't work well.

    def start_scenario(self, scenario_name="Margit"):
        """
        Start the scenario with the given scenario name.
        """
        # FIXME: This is a hack to start boss fight. Need to check fogwall state. or use another method.
        x, y, z = self.scenarios[scenario_name]["fog_wall_location"]
        self.teleport(x, y, z)
        self.move_mouse(1000, 0, 1)
        sleep(2)
        self.send_key(["W", "E"], 200, 200)
        sleep(2)
        self.send_key(["B"], 200)
