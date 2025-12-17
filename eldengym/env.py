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
        keybinds_filepath (str): Path to keybinds JSON file
        siphon_config_filepath (str): Path to siphon TOML config
        memory_attributes (list[str]): List of memory attribute names to include in observation.
            Default: ["HeroHp", "HeroMaxHp", "NpcHp", "NpcMaxHp", "HeroAnimId", "NpcAnimId"]
        host (str): Siphon server host. Default: 'localhost:50051'
        reward_function (RewardFunction): Custom reward function
        frame_format (str): Frame format for streaming ('jpeg' or 'raw'). Default: 'jpeg'
        frame_quality (int): JPEG quality 1-100. Default: 85
        max_steps (int): Maximum steps per episode. Default: None
    """

    def __init__(
        self,
        scenario_name,
        keybinds_filepath,
        siphon_config_filepath,
        memory_attributes=None,
        host="localhost:50051",
        reward_function=None,
        frame_format="jpeg",
        frame_quality=85,
        max_steps=None,
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

        # Memory attributes to poll (configurable, not hardcoded)
        self.memory_attributes = memory_attributes or [
            "HeroHp",
            "HeroMaxHp",
            "NpcHp",
            "NpcMaxHp",
            "HeroAnimId",
            "NpcAnimId",
        ]

        # Load keybinds
        with open(self.keybinds_filepath, "r") as f:
            keybinds_data = json.load(f)
            self.keybinds = keybinds_data["keybinds"]

        # Create action space (multi-binary for all keys)
        self.action_keys = list(self.keybinds.keys())
        self.action_space = gym.spaces.MultiBinary(len(self.action_keys))

        # Track current key states for toggling
        self._key_states = {key: False for key in self.action_keys}

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
        print("Launching game...")
        self.client.launch_game()
        time.sleep(20)  # Wait for game to launch

        print("Initializing Siphon...")
        self.client.load_config_from_file(self.siphon_config_filepath, wait_time=2)
        time.sleep(2)

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
        frame = self.client.get_latest_frame(self._stream_handle)

        # If no new frame available, wait briefly and retry
        if frame is None:
            time.sleep(0.005)
            frame = self.client.get_latest_frame(self._stream_handle)

        # Get memory attributes
        memory_data = {}
        for attr_name in self.memory_attributes:
            try:
                memory_data[attr_name] = self.client.get_attribute(attr_name)
            except Exception as e:
                print(f"Warning: Could not read attribute {attr_name}: {e}")
                memory_data[attr_name] = 0

        # Combine into observation
        obs = {"frame": frame, **memory_data}

        return obs

    def _toggle_keys(self, action):
        """
        Toggle keys based on multi-binary action and current key states.

        Args:
            action: Multi-binary array indicating desired key states
        """
        for i, desired_state in enumerate(action):
            key = self.action_keys[i]
            current_state = self._key_states[key]
            new_state = bool(desired_state)

            # Only toggle if state changed
            if new_state != current_state:
                self.client.input_key_toggle(key, new_state)
                self._key_states[key] = new_state

    def _release_all_keys(self):
        """Release all currently pressed keys."""
        for key, is_pressed in self._key_states.items():
            if is_pressed:
                self.client.input_key_toggle(key, False)
                self._key_states[key] = False

    def reset(self, seed=None, options=None):
        """Reset environment - start new episode."""
        super().reset(seed=seed)

        # Release all keys from previous episode
        self._release_all_keys()

        # Reset game state (implement based on your needs)
        # TODO: Implement proper reset logic
        # For now, just wait briefly
        time.sleep(1)

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
                    **{
                        attr: gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                        )
                        for attr in self.memory_attributes
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
            action: Multi-binary array [0/1] for each key in self.action_keys

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
