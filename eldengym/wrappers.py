import gymnasium as gym
import numpy as np
from collections import deque


class DictFrameStack(gym.ObservationWrapper):
    """
    Stack last N frames for Dict observation spaces.

    Stacks the 'frame' key while preserving other observation keys.

    Args:
        env: Environment with Dict observation space
        num_stack: Number of frames to stack (default: 4)
        frame_key: Key for frame data in observation dict (default: 'frame')
    """

    def __init__(self, env, num_stack=4, frame_key="frame"):
        super().__init__(env)
        self.num_stack = num_stack
        self.frame_key = frame_key
        self.frames = deque(maxlen=num_stack)

        # Update observation space - modify only the frame space
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("DictFrameStack requires Dict observation space")

        if frame_key not in self.observation_space.spaces:
            raise ValueError(f"Frame key '{frame_key}' not found in observation space")

        # Get original frame space
        frame_space = self.observation_space.spaces[frame_key]

        # Create stacked frame space
        if len(frame_space.shape) == 3:  # (H, W, C)
            new_shape = (*frame_space.shape[:2], frame_space.shape[2] * num_stack)
        else:
            raise ValueError(f"Unexpected frame shape: {frame_space.shape}")

        # Update observation space with stacked frames
        new_spaces = self.observation_space.spaces.copy()
        new_spaces[frame_key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=frame_space.dtype,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        """Stack frames and return modified observation."""
        frame = obs[self.frame_key]
        self.frames.append(frame)

        # Pad with first frame if not enough frames yet
        while len(self.frames) < self.num_stack:
            self.frames.append(frame)

        # Stack frames along channel dimension
        stacked_frame = np.concatenate(list(self.frames), axis=-1)

        # Return modified observation with stacked frame
        obs_copy = obs.copy()
        obs_copy[self.frame_key] = stacked_frame
        return obs_copy

    def reset(self, **kwargs):
        """Reset and clear frame buffer."""
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        return self.observation(obs), info


class DictResizeFrame(gym.ObservationWrapper):
    """
    Resize frames in Dict observation spaces.

    Args:
        env: Environment with Dict observation space
        width: Target width (default: 84)
        height: Target height (default: 84)
        frame_key: Key for frame data in observation dict (default: 'frame')
    """

    def __init__(self, env, width=84, height=84, frame_key="frame"):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_key = frame_key

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("DictResizeFrame requires Dict observation space")

        # Update frame space with new dimensions
        frame_space = self.observation_space.spaces[frame_key]
        new_spaces = self.observation_space.spaces.copy()
        new_spaces[frame_key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, frame_space.shape[-1]),
            dtype=frame_space.dtype,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        """Resize frame and return modified observation."""
        import cv2

        frame = obs[self.frame_key]
        resized_frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )

        obs_copy = obs.copy()
        obs_copy[self.frame_key] = resized_frame
        return obs_copy


class DictGrayscaleFrame(gym.ObservationWrapper):
    """
    Convert frames to grayscale in Dict observation spaces.

    Args:
        env: Environment with Dict observation space
        frame_key: Key for frame data in observation dict (default: 'frame')
    """

    def __init__(self, env, frame_key="frame"):
        super().__init__(env)
        self.frame_key = frame_key

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("DictGrayscaleFrame requires Dict observation space")

        # Update frame space for grayscale
        frame_space = self.observation_space.spaces[frame_key]
        new_spaces = self.observation_space.spaces.copy()
        new_spaces[frame_key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_space.shape[0], frame_space.shape[1], 1),
            dtype=frame_space.dtype,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        """Convert frame to grayscale and return modified observation."""
        import cv2

        frame = obs[self.frame_key]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, -1)

        obs_copy = obs.copy()
        obs_copy[self.frame_key] = gray
        return obs_copy


class NormalizeMemoryAttributes(gym.ObservationWrapper):
    """
    Normalize memory attribute values to [0, 1] or [-1, 1].

    Args:
        env: Environment with Dict observation space
        attribute_ranges: Dict mapping attribute names to (min, max) tuples.
            If not provided, will use observed min/max during runtime.
        frame_key: Key to skip normalization (default: 'frame')
    """

    def __init__(self, env, attribute_ranges=None, frame_key="frame"):
        super().__init__(env)
        self.frame_key = frame_key
        self.attribute_ranges = attribute_ranges or {}

        # Track observed ranges for adaptive normalization
        self.observed_min = {}
        self.observed_max = {}

    def observation(self, obs):
        """Normalize memory attributes."""
        obs_copy = obs.copy()

        for key, value in obs.items():
            # Skip frame data
            if key == self.frame_key:
                continue

            # Get or update range
            if key in self.attribute_ranges:
                min_val, max_val = self.attribute_ranges[key]
            else:
                # Track observed range
                if key not in self.observed_min:
                    self.observed_min[key] = value
                    self.observed_max[key] = value
                else:
                    self.observed_min[key] = min(self.observed_min[key], value)
                    self.observed_max[key] = max(self.observed_max[key], value)

                min_val = self.observed_min[key]
                max_val = self.observed_max[key]

            # Normalize to [0, 1]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.0

            obs_copy[key] = normalized

        return obs_copy


# Legacy wrappers for backward compatibility (simple array observations)
class FrameStack(gym.ObservationWrapper):
    """Stack last N frames (for simple array observations)"""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Update observation space
        low = np.repeat(self.observation_space.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(
            self.observation_space.high[..., np.newaxis], num_stack, axis=-1
        )

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        self.frames.append(obs)
        # Pad with first frame if not enough frames yet
        while len(self.frames) < self.num_stack:
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        return self.observation(obs), info


class ResizeFrame(gym.ObservationWrapper):
    """Resize frames to target shape (for simple array observations)"""

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, self.observation_space.shape[-1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        import cv2

        return cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)


class GrayscaleFrame(gym.ObservationWrapper):
    """Convert to grayscale (for simple array observations)"""

    def __init__(self, env):
        super().__init__(env)

        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(old_shape[0], old_shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, -1)


class HPRefundWrapper(gym.Wrapper):
    """
    Refund player and/or boss HP after each step.

    Useful for evaluation and data collection where you want to prevent
    episode termination due to HP loss.

    Tracks damage by comparing consecutive observations to avoid race
    conditions with the refund timing.

    Args:
        env: EldenGym environment
        refund_player: Whether to refund player HP (default: True)
        refund_boss: Whether to refund boss HP (default: False)
        player_hp_attr: Attribute name for player HP (default: 'HeroHp')
        player_max_hp_attr: Attribute name for player max HP (default: 'HeroMaxHp')
        boss_hp_attr: Attribute name for boss HP (default: 'NpcHp')
        boss_max_hp_attr: Attribute name for boss max HP (default: 'NpcMaxHp')
    """

    def __init__(
        self,
        env,
        refund_player=True,
        refund_boss=False,
        player_hp_attr="HeroHp",
        player_max_hp_attr="HeroMaxHp",
        boss_hp_attr="NpcHp",
        boss_max_hp_attr="NpcMaxHp",
    ):
        super().__init__(env)
        self.refund_player = refund_player
        self.refund_boss = refund_boss
        self.player_hp_attr = player_hp_attr
        self.player_max_hp_attr = player_max_hp_attr
        self.boss_hp_attr = boss_hp_attr
        self.boss_max_hp_attr = boss_max_hp_attr

        # Track previous HP to calculate damage delta
        self._prev_player_hp = None
        self._prev_boss_hp = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize HP tracking from first observation
        self._prev_player_hp = obs.get(self.player_hp_attr, 0)
        self._prev_boss_hp = obs.get(self.boss_hp_attr, 0)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track player damage using delta from previous HP
        if self.refund_player:
            max_hp = obs.get(self.player_max_hp_attr, 0)
            current_hp = obs.get(self.player_hp_attr, 0)

            if max_hp > 0:
                # Calculate damage as drop from previous HP
                if self._prev_player_hp is not None:
                    player_damage = max(0, self._prev_player_hp - current_hp)
                else:
                    player_damage = 0

                info["player_damage_taken"] = player_damage
                info["player_damage_taken_normalized"] = player_damage / max_hp

                # Only refund if damage was taken
                if player_damage > 0:
                    self.env.client.set_attribute(self.player_hp_attr, int(max_hp), "int")
                    self._prev_player_hp = max_hp
                else:
                    self._prev_player_hp = current_hp

        # Track boss damage using delta from previous HP
        if self.refund_boss:
            max_hp = obs.get(self.boss_max_hp_attr, 0)
            current_hp = obs.get(self.boss_hp_attr, 0)

            if max_hp > 0:
                # Calculate damage as drop from previous HP
                if self._prev_boss_hp is not None:
                    boss_damage = max(0, self._prev_boss_hp - current_hp)
                else:
                    boss_damage = 0

                info["boss_damage_dealt"] = boss_damage
                info["boss_damage_dealt_normalized"] = boss_damage / max_hp

                # Only refund if damage was dealt
                if boss_damage > 0:
                    self.env.client.set_attribute(self.boss_hp_attr, int(max_hp), "int")
                    self._prev_boss_hp = max_hp
                else:
                    self._prev_boss_hp = current_hp

        return obs, reward, terminated, truncated, info
