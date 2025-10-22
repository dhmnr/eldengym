import gymnasium as gym
from .client.elden_client import EldenClient
from .rewards import RewardFunction, ScoreDeltaReward
from time import sleep
from enum import Enum


class ActionType(Enum):
    """Categorize actions by their properties"""

    INSTANT = "instant"  # No-op
    MOVEMENT = "movement"  # Walking, running
    COMBAT = "combat"  # Attacks, spells
    DODGE = "dodge"  # Rolling, jumping


class EldenGymEnv(gym.Env):
    """
    Elden Ring Gymnasium environment - OpenAI Five style approach.

    Simple design philosophy:
    - Fixed timestep observations (like Dota/Atari)
    - Agent gets current game state every step
    - Let RL figure out action timing from animation_id

    Args:
        scenario_name (str): Boss scenario. Default: 'margit'
        host (str): Siphon client host. Default: 'localhost:50051'
        action_mode (str): Action space type.
            - 'discrete': Single action per step (14 actions)
            - 'multi_binary': Multiple keys per step (11 keys)
        reward_function (RewardFunction): Custom reward function.
        frame_skip (int): Frames to skip between observations.
            Higher = faster but less responsive. Default: 4
        game_speed (float): Game speed (0.1-1.0).
            Lower = easier for agent. Default: 1.0
        freeze_game (bool): Whether to freeze the game. Default: False
        game_fps (int): Game FPS. Default: 60
        max_step (int): Maximum number of steps. Default: None (infinite horizon)

    Action Spaces:
        discrete: Gym.spaces.Discrete(14)
            - Predefined combos (attack, dodge_forward, etc.)
        multi_binary: Gym.spaces.MultiBinary(11)
            - Individual keys (W, A, S, D, SPACE, SHIFT, etc.)
            - Agent can press multiple keys simultaneously
            - More flexible, closer to human input
    """

    def __init__(
        self,
        scenario_name="margit",
        host="localhost:50051",
        action_mode="discrete",
        reward_function=None,
        frame_skip=4,  # Number of frames to skip (like Atari)
        game_speed=1.0,  # Game speed multiplier
        freeze_game=False,
        game_fps=60,
        max_step=None,
        config_filepath="ER_1_16_1.toml",  # Auto-resolves to eldengym/files/configs/
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.action_mode = action_mode
        self.frame_skip = frame_skip
        self.game_speed = game_speed
        self.freeze_game = freeze_game
        self.game_fps = game_fps
        self.max_step = max_step
        self.config_filepath = config_filepath
        self.step_count = 0
        # Reward function
        self.reward_function = reward_function or ScoreDeltaReward(
            score_key="player_hp"
        )
        if not isinstance(self.reward_function, RewardFunction):
            raise TypeError("reward_fn must inherit from RewardFunction")

        # Actions
        if action_mode == "discrete":
            self.action_map = self._discrete_action_map()
            self.action_space = gym.spaces.Discrete(len(self.action_map))
            self.action_keybindings = self._discrete_action_keybindings()
        elif action_mode == "multi_binary":
            self.action_map = self._multi_binary_action_map()
            self.action_space = gym.spaces.MultiBinary(len(self.action_map))
            self.action_keybindings = None  # Not used in multi-binary
        else:
            raise ValueError(f"Invalid action mode: {action_mode}")

        # Simple state tracking
        self._prev_info = None
        self._last_animation_id = None

        # self._download_savefile() #TODO: Implement this
        self.client.launch_game()
        sleep(20)  # Wait for game to launch
        self.client.load_config_from_file(self.config_filepath, wait_time=2)
        sleep(2)  # Wait for config to load
        self.client.bypass_menu()
        sleep(10)  # Wait for game to load
        self.first_load = True

    def _multi_binary_action_map(self):
        """
        Multi-binary action mapping - OpenAI Five style.

        Each index can be 0 or 1 (pressed or not).
        Agent can press multiple keys simultaneously.
        """
        return {
            0: "W",  # Forward
            1: "A",  # Left
            2: "S",  # Backward
            3: "D",  # Right
            4: "SPACE",  # Jump
            5: "LEFT_SHIFT",  # Dodge/Sprint
            6: "E",  # Interact
            7: "LEFT_ALT",  # Heavy attack modifier key
            8: "R",  # Use item
            9: "F",  # Weapon art
            10: "LEFT",  # Attack key
        }

    def _discrete_action_map(self):
        """Action mapping"""
        return {
            0: {"name": "no-op", "type": ActionType.INSTANT},
            1: {"name": "forward", "type": ActionType.MOVEMENT},
            2: {"name": "backward", "type": ActionType.MOVEMENT},
            3: {"name": "left", "type": ActionType.MOVEMENT},
            4: {"name": "right", "type": ActionType.MOVEMENT},
            5: {"name": "jump", "type": ActionType.DODGE},
            6: {"name": "dodge_forward", "type": ActionType.DODGE},
            7: {"name": "dodge_backward", "type": ActionType.DODGE},
            8: {"name": "dodge_left", "type": ActionType.DODGE},
            9: {"name": "dodge_right", "type": ActionType.DODGE},
            10: {"name": "interact", "type": ActionType.INSTANT},
            11: {"name": "attack", "type": ActionType.COMBAT},
            12: {"name": "use_item", "type": ActionType.COMBAT},
            13: {"name": "weapon_art", "type": ActionType.COMBAT},
        }

    def _discrete_action_keybindings(self):
        """Keybindings for each action"""
        return {
            "no-op": [],
            "forward": [["W"], 500, 0],
            "backward": [["S"], 500, 0],
            "left": [["A"], 500, 0],
            "right": [["D"], 500, 0],
            "jump": [["SPACE"], 500, 0],
            "dodge_forward": [["W", "LEFT_SHIFT"], 100, 200],
            "dodge_backward": [["S", "LEFT_SHIFT"], 100, 200],
            "dodge_left": [["A", "LEFT_SHIFT"], 100, 200],
            "dodge_right": [["D", "LEFT_SHIFT"], 100, 200],
            "interact": [["E"], 500, 0],
            "attack": [["LEFT_ALT", "LEFT"], 400, 0],
            "use_item": [["R"], 500, 0],
            "weapon_art": [["F"], 500, 0],
        }

    def reset(self, seed=None, options=None):
        """Reset environment - start new episode."""
        super().reset(seed=seed)

        # Set game speed
        self.client.set_game_speed(self.game_speed)

        # Reset game and start scenario
        if self.first_load:
            self.first_load = False
        else:
            self.client.reset_game()

        self.client.start_scenario(self.scenario_name)
        sleep(1)  # Wait for fight to start

        # Reset state
        self._prev_info = None
        self._last_animation_id = self.client.player_animation_id

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        self._prev_info = info.copy()

        if self.freeze_game:
            self.client.set_game_speed(1e-5)

        return obs, info

    def step(self, action):
        """
        Execute one step

        send action, wait frame_skip frames, return observation.

        Args:
            action: int (discrete mode) or array (multi_binary mode)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        if self.freeze_game:
            self.client.set_game_speed(self.game_speed)

        # Send action based on mode

        if self.action_mode == "discrete":
            # Discrete: single action ID
            action_data = self.action_map[action]
            action_name = action_data["name"]

            if action_name != "no-op":
                keybinding = self.action_keybindings.get(action_name, [])
                if keybinding:
                    self.client.send_key(*keybinding)

        elif self.action_mode == "multi_binary":
            # Multi-binary: array of 0s and 1s
            # Collect all keys that should be pressed (where action[i] == 1)
            keys_to_press = []
            for i, should_press in enumerate(action):
                if should_press == 1:
                    keys_to_press.append(self.action_map[i])

            # Send all keys simultaneously if any
            if keys_to_press:
                self.client.send_key(keys_to_press, 100, 0)  # Press all together

        # Wait frame_skip frames (like Atari)
        # Game runs at ~60fps, so frame_skip=4 means ~0.067s between observations
        sleep(self.frame_skip / self.game_fps)

        if self.freeze_game:
            self.client.set_game_speed(1e-5)

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        # Calculate reward
        reward = self.reward_function.calculate(obs, info, self._prev_info)

        # Check termination
        terminated = self.reward_function.is_done(obs, info)
        truncated = (
            self.step_count >= self.max_step if self.max_step is not None else False
        )

        if terminated or truncated:
            self.step_count = 0
        else:
            self.step_count += 1

        # Update state
        self._prev_info = info.copy()
        self._last_animation_id = self.client.player_animation_id

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get raw observation - let agent figure out timing.

        Returns:
            dict: Observation with:
                - frame: Game frame/image
                - boss_hp: Boss health (0.0-1.0)
                - player_hp: Player health (0.0-1.0)
                - distance: Distance to boss
                - boss_animation_id: Boss animation (for predicting attacks)
                - player_animation_id: Player animation (for knowing if stuck in animation)
                - last_animation_id: Previous player animation (to detect changes)
        """
        frame = self.client.get_frame()

        return {
            "frame": frame,
            "boss_hp": self.client.target_hp / self.client.target_max_hp,
            "player_hp": self.client.player_hp / self.client.player_max_hp,
            "distance": self.client.target_player_distance,
            "boss_animation_id": self.client.target_animation_id,
            "player_animation_id": self.client.player_animation_id,
            "last_animation_id": self._last_animation_id,
        }

    def _get_info(self):
        """Extra debug info"""
        return {
            "player_hp": self.client.player_hp,
            "boss_hp": self.client.target_hp,
        }

    def close(self):
        """
        Close the environment and clean up resources.

        This method closes the connection to the siphon client.
        """
        self.client.close()
