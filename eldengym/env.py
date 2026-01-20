import gymnasium as gym
import numpy as np
import json
import time
from .client.elden_client import EldenClient
from .rewards import RewardFunction, ScoreDeltaReward


class EldenGymEnv(gym.Env):
    """
    Elden Ring Gymnasium environment with non-blocking frame streaming.

    Uses pysiphon's frame streaming for efficient polling-based observations.

    Args:
        scenario_name (str): Boss scenario name
        keybinds_filepath (str): Path to keybinds JSON file (v2 format: action → keys)
        siphon_config_filepath (str): Path to siphon TOML config
        memory_attributes (list[str]): List of memory attribute names to include in observation.
            Default: ["HeroHp", "HeroMaxHp", "NpcHp", "NpcMaxHp", "HeroAnimId", "NpcAnimId"]
        actions (list[str], optional): List of action names to include in action space.
            If None, all actions from keybinds file are used.
            Example: ["move_forward", "move_back", "move_left", "move_right", "dodge_roll/dash"]
        host (str): Siphon server host. Default: 'localhost:50051'
        reward_function (RewardFunction): Custom reward function
        frame_format (str): Frame format for streaming ('jpeg' or 'raw'). Default: 'jpeg'
        frame_quality (int): JPEG quality 1-100. Default: 85
        max_steps (int): Maximum steps per episode. Default: None
        launch_game (bool): Whether to launch the game automatically. Default: True
            Set to False if game is already running
        save_file_name (str, optional): Name of backup save file to copy during reset
            (e.g., "margit_checkpoint.sl2"). If None, no save file copying occurs.
        save_file_dir (str, optional): Directory containing save files. Required if
            save_file_name is provided. Typically: %APPDATA%/EldenRing/<steam_id>/
        use_device (str): Preferred input device - 'key' for keyboard (default) or 'mouse'.
            When 'mouse' is selected, uses mouse binding if available, otherwise falls back to keyboard.
    """

    def __init__(
        self,
        scenario_name,
        keybinds_filepath,
        siphon_config_filepath,
        memory_attributes=None,
        actions=None,
        host="localhost:50051",
        reward_function=None,
        frame_format="jpeg",
        frame_quality=85,
        max_steps=None,
        launch_game=True,
        save_file_name=None,
        save_file_dir=None,
        use_device="key",
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.keybinds_filepath = keybinds_filepath
        self.siphon_config_filepath = siphon_config_filepath
        self.step_count = 0
        self.max_steps = max_steps
        self.frame_format = frame_format
        self.frame_quality = frame_quality
        self.save_file_name = save_file_name
        self.save_file_dir = save_file_dir
        self.use_device = use_device

        # Memory attributes to poll (configurable, not hardcoded)
        self.memory_attributes = memory_attributes or [
            "HeroHp",
            "HeroMaxHp",
            "NpcHp",
            "NpcMaxHp",
            "HeroAnimId",
            "NpcAnimId",
        ]

        # Coordinate attributes (always polled for real coords computation)
        self._coord_attributes = [
            "HeroGlobalPosX",
            "HeroGlobalPosY",
            "HeroGlobalPosZ",
            "HeroLocalPosX",
            "HeroLocalPosY",
            "HeroLocalPosZ",
            "NpcGlobalPosX",  # Actually local coords, needs transform
            "NpcGlobalPosY",
            "NpcGlobalPosZ",
        ]

        # Real coord attribute names (added to obs)
        self._real_coord_attrs = [
            "player_x",
            "player_y",
            "player_z",
            "boss_x",
            "boss_y",
            "boss_z",
            "dist_to_boss",
            "boss_z_relative",
        ]

        # Load keybinds (v2 format: action → keys with index)
        with open(self.keybinds_filepath, "r") as f:
            keybinds_data = json.load(f)
            all_action_bindings = keybinds_data["actions"]

        # Filter actions if specified
        if actions is not None:
            # Validate requested actions exist
            invalid_actions = set(actions) - set(all_action_bindings.keys())
            if invalid_actions:
                raise ValueError(
                    f"Unknown actions: {invalid_actions}. "
                    f"Available: {list(all_action_bindings.keys())}"
                )
            # Filter to only requested actions, preserving order from actions list
            self._action_bindings = {a: all_action_bindings[a] for a in actions}
            # Use order from actions parameter
            sorted_actions = [(a, self._action_bindings[a]) for a in actions]
        else:
            self._action_bindings = all_action_bindings
            # Sort actions by index to ensure consistent ordering
            sorted_actions = sorted(
                self._action_bindings.items(), key=lambda x: x[1]["index"]
            )

        # Build action-to-key mapping based on use_device preference
        self._action_to_key = {}
        for action, bindings in sorted_actions:
            if self.use_device == "mouse" and "mouse" in bindings:
                self._action_to_key[action] = bindings["mouse"]
            else:
                self._action_to_key[action] = bindings["key"]

        # Create action space (multi-binary for selected actions)
        # action_keys preserves the order (index in MultiBinary = position in this list)
        self.action_keys = [action for action, _ in sorted_actions]
        self.action_space = gym.spaces.MultiBinary(len(self.action_keys))

        # Track current key states for toggling (using actual keys, not actions)
        self._active_keys = set(self._action_to_key.values())
        self._key_states = {key: False for key in self._active_keys}

        # Frame stream handle
        self._stream_handle = None

        # Reward function
        self.reward_function = reward_function or ScoreDeltaReward(
            score_key="player_hp"
        )
        if not isinstance(self.reward_function, RewardFunction):
            raise TypeError("reward_fn must inherit from RewardFunction")

        # State tracking
        self._prev_info = None

        # Initialize game and siphon
        if launch_game:
            print("Launching game...")
            self.client.launch_game()
            time.sleep(20)  # Wait for game to launch
            self.client.enter_game()
        else:
            print("Skipping game launch (launch_game=False)")

        print("Initializing Siphon...")
        self.client.load_config_from_file(self.siphon_config_filepath, wait_time=2)
        time.sleep(2)

        # Verify server is ready
        print("Checking server status...")
        status = self.client.get_server_status()
        print(f"Server status: {status}")

        if not status.get("memory_initialized", False):
            raise RuntimeError("Memory subsystem not initialized!")
        if not status.get("capture_initialized", False):
            raise RuntimeError("Capture subsystem not initialized!")

        print("Starting frame stream...")
        self._stream_handle = self.client.start_frame_stream(
            format=self.frame_format, quality=self.frame_quality
        )

        # Setup observation space (will be defined after first observation)
        self.observation_space = None

    def _poll_observation(self):
        """
        Poll for latest frame and memory attributes.

        Returns:
            dict: Observation with 'frame' and memory attributes
        """
        # Poll latest frame (non-blocking)
        frame_data = self.client.get_latest_frame(self._stream_handle)

        # If no new frame available, wait briefly and retry
        if frame_data is None:
            time.sleep(0.005)
            frame_data = self.client.get_latest_frame(self._stream_handle)

        # Decode frame from protobuf FrameData object
        if frame_data is not None:
            import cv2

            # Extract JPEG bytes from protobuf
            jpeg_bytes = frame_data.data

            if jpeg_bytes and len(jpeg_bytes) > 0:
                # Decode JPEG bytes to numpy array
                frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                # Fallback: create black frame if no data
                frame = np.zeros(
                    (frame_data.height, frame_data.width, 3), dtype=np.uint8
                )
        else:
            # No frame available, create black placeholder
            frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        # Get memory attributes
        memory_data = {}
        for attr_name in self.memory_attributes:
            try:
                response = self.client.get_attribute(attr_name)
                # pysiphon returns dict with 'value' key
                if isinstance(response, dict):
                    value = response.get("value", 0)
                else:
                    value = response
                memory_data[attr_name] = value
            except Exception as e:
                print(f"Warning: Could not read attribute {attr_name}: {e}")
                print(
                    "  This might mean the game isn't fully loaded yet or the attribute doesn't exist."
                )
                memory_data[attr_name] = 0

        # Get coordinate attributes for real coords computation
        coord_data = {}
        for attr_name in self._coord_attributes:
            try:
                response = self.client.get_attribute(attr_name)
                if isinstance(response, dict):
                    value = response.get("value", 0)
                else:
                    value = response
                coord_data[attr_name] = value
            except Exception:
                coord_data[attr_name] = 0

        # Compute real coordinates
        real_coords = self._compute_real_coords(coord_data)

        # Combine into observation
        obs = {"frame": frame, **memory_data, **real_coords}

        return obs

    def _compute_real_coords(self, coord_data):
        """
        Compute real world coordinates from raw coordinate data.

        Player uses HeroGlobalPos directly (truly global).
        Boss uses transform: npc_local + (hero_global - hero_local).

        Args:
            coord_data: Dict with raw coordinate attributes

        Returns:
            Dict with player_x/y/z, boss_x/y/z, dist_to_boss, boss_z_relative
        """
        # Player coords (HeroGlobalPos is truly global)
        player_x = coord_data.get("HeroGlobalPosX", 0)
        player_y = coord_data.get("HeroGlobalPosY", 0)
        player_z = coord_data.get("HeroGlobalPosZ", 0)

        # Compute local→global transform
        hero_local_x = coord_data.get("HeroLocalPosX", 0)
        hero_local_y = coord_data.get("HeroLocalPosY", 0)
        hero_local_z = coord_data.get("HeroLocalPosZ", 0)

        transform_x = player_x - hero_local_x
        transform_y = player_y - hero_local_y
        transform_z = player_z - hero_local_z

        # Boss coords (NpcGlobalPos is actually local, apply transform)
        npc_local_x = coord_data.get("NpcGlobalPosX", 0)
        npc_local_y = coord_data.get("NpcGlobalPosY", 0)
        npc_local_z = coord_data.get("NpcGlobalPosZ", 0)

        boss_x = npc_local_x + transform_x
        boss_y = npc_local_y + transform_y
        boss_z = npc_local_z + transform_z

        # Derived values
        dx = player_x - boss_x
        dy = player_y - boss_y
        dist_to_boss = np.sqrt(dx * dx + dy * dy)  # XY distance only
        boss_z_relative = boss_z - player_z

        return {
            "player_x": player_x,
            "player_y": player_y,
            "player_z": player_z,
            "boss_x": boss_x,
            "boss_y": boss_y,
            "boss_z": boss_z,
            "dist_to_boss": dist_to_boss,
            "boss_z_relative": boss_z_relative,
        }

    def _toggle_keys(self, action):
        """
        Toggle keys based on multi-binary action and current key states.

        Translates semantic actions to actual keyboard/mouse keys.

        Args:
            action: Multi-binary array indicating desired action states
        """
        for i, desired_state in enumerate(action):
            semantic_action = self.action_keys[i]
            key = self._action_to_key[semantic_action]
            current_state = self._key_states[key]
            new_state = bool(desired_state)

            # Only toggle if state changed
            if new_state != current_state:
                self.client.input_key_toggle(key, new_state)
                self._key_states[key] = new_state

    def _release_all_keys(self):
        """Release all currently pressed keys."""
        for key in self._active_keys:
            if self._key_states.get(key, False):
                self.client.input_key_toggle(key, False)
                self._key_states[key] = False

    def reset(self, seed=None, options=None):
        """Reset environment - start new episode."""
        super().reset(seed=seed)

        # Release all keys from previous episode
        self._release_all_keys()

        # Reset game state
        self.client.quit_to_title()

        # Copy save file if configured
        if self.save_file_name and self.save_file_dir:
            print(f"Copying save file: {self.save_file_name}")
            self.client.copy_save_file(self.save_file_name, self.save_file_dir)

        self.client.enter_menu()
        self.client.start_scenario(self.scenario_name)

        # Reset tracking
        self.step_count = 0
        self._prev_info = None

        # Get initial observation
        obs = self._poll_observation()

        # Define observation space on first reset if not already defined
        if self.observation_space is None:
            self.observation_space = gym.spaces.Dict(
                {
                    "frame": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=obs["frame"].shape,
                        dtype=np.uint8,
                    ),
                    # User-configured memory attributes
                    **{
                        attr: gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                        )
                        for attr in self.memory_attributes
                    },
                    # Real coordinate attributes (always included)
                    **{
                        attr: gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                        )
                        for attr in self._real_coord_attrs
                    },
                }
            )

        info = self._get_info(obs)
        self._prev_info = info.copy()

        return obs, info

    def step(self, action):
        """
        Execute one step with key toggling.

        Args:
            action: Multi-binary array [0/1] for each semantic action in self.action_keys
                e.g., [1, 0, 0, 1, ...] to activate move_forward and dodge_roll/dash

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Toggle keys based on action
        self._toggle_keys(action)

        # Brief wait for game to process input
        time.sleep(0.016)  # ~1 frame at 60fps

        # Poll observation
        obs = self._poll_observation()
        info = self._get_info(obs)

        # Calculate reward
        reward = self.reward_function.calculate(obs, info, self._prev_info)

        # Check termination
        terminated = self.reward_function.is_done(obs, info)
        truncated = (
            self.step_count >= self.max_steps if self.max_steps is not None else False
        )

        # Update tracking
        self.step_count += 1
        self._prev_info = info.copy()

        return obs, reward, terminated, truncated, info

    def _get_info(self, obs):
        """
        Extract info dict from observation.

        Args:
            obs: Observation dict

        Returns:
            dict: Info with normalized/processed values
        """
        info = {}

        # Add normalized HP values if available
        if "HeroHp" in obs and "HeroMaxHp" in obs:
            info["player_hp_normalized"] = (
                obs["HeroHp"] / obs["HeroMaxHp"] if obs["HeroMaxHp"] > 0 else 0
            )

        if "NpcHp" in obs and "NpcMaxHp" in obs:
            info["boss_hp_normalized"] = (
                obs["NpcHp"] / obs["NpcMaxHp"] if obs["NpcMaxHp"] > 0 else 0
            )

        # Add animation IDs
        if "HeroAnimId" in obs:
            info["player_animation"] = obs["HeroAnimId"]

        if "NpcAnimId" in obs:
            info["boss_animation"] = obs["NpcAnimId"]

        # Add real coords as tuples for convenience (debugging)
        info["player_xyz"] = (
            obs.get("player_x", 0),
            obs.get("player_y", 0),
            obs.get("player_z", 0),
        )
        info["boss_xyz"] = (
            obs.get("boss_x", 0),
            obs.get("boss_y", 0),
            obs.get("boss_z", 0),
        )

        return info

    def close(self):
        """Close environment and clean up resources."""
        # Stop frame stream
        if self._stream_handle is not None:
            self.client.stop_frame_stream(self._stream_handle)
            self._stream_handle = None

        # Release all keys
        self._release_all_keys()

        # Close client
        self.client.close()

    def render(self):
        """Render is handled by the game itself."""
        pass
