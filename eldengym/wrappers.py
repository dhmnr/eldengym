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
                    self.unwrapped.client.set_attribute(self.player_hp_attr, int(max_hp), "int")
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
                    self.unwrapped.client.set_attribute(self.boss_hp_attr, int(max_hp), "int")
                    self._prev_boss_hp = max_hp
                else:
                    self._prev_boss_hp = current_hp

        return obs, reward, terminated, truncated, info


class AnimFrameWrapper(gym.Wrapper):
    """
    Track boss animation ID and elapsed frames with same animation.

    Adds to observation:
    - boss_anim_id: Current NpcAnimId
    - elapsed_frames: Number of frames since animation changed

    Args:
        env: EldenGym environment
        anim_id_key: Key for animation ID in obs (default: 'NpcAnimId')
    """

    def __init__(self, env, anim_id_key="NpcAnimId"):
        super().__init__(env)
        self.anim_id_key = anim_id_key
        self._prev_anim_id = None
        self._elapsed_frames = 0

        # Extend observation space
        if hasattr(self.env, 'observation_space') and self.env.observation_space is not None:
            self._extend_obs_space()

    def _extend_obs_space(self):
        """Extend observation space with new attributes."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["boss_anim_id"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(), dtype=np.float32
            )
            new_spaces["elapsed_frames"] = gym.spaces.Box(
                low=0, high=np.inf, shape=(), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Extend obs space on first reset if needed
        if not hasattr(self, '_obs_space_extended'):
            self._extend_obs_space()
            self._obs_space_extended = True

        # Initialize tracking
        self._prev_anim_id = obs.get(self.anim_id_key, 0)
        self._elapsed_frames = 0

        # Add to obs
        obs["boss_anim_id"] = self._prev_anim_id
        obs["elapsed_frames"] = self._elapsed_frames

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_anim_id = obs.get(self.anim_id_key, 0)

        # Check if animation changed
        if current_anim_id != self._prev_anim_id:
            self._elapsed_frames = 0
            self._prev_anim_id = current_anim_id
        else:
            self._elapsed_frames += 1

        # Add to obs
        obs["boss_anim_id"] = current_anim_id
        obs["elapsed_frames"] = self._elapsed_frames

        return obs, reward, terminated, truncated, info


class SDFObsWrapper(gym.Wrapper):
    """
    Add SDF (Signed Distance Field) observations for arena boundary awareness.

    Adds to observation:
    - sdf_value: Signed distance to boundary (negative = inside)
    - sdf_normal_x: X component of normal vector pointing to boundary
    - sdf_normal_y: Y component of normal vector pointing to boundary

    Requires player coordinates in obs (player_x, player_y from EldenGymEnv).

    Args:
        env: EldenGym environment with real coords
        boundary: ArenaBoundary instance with query_sdf(x, y) method
        live_plot: Enable live visualization of positions and SDF
    """

    def __init__(self, env, boundary, live_plot=False):
        super().__init__(env)
        self.boundary = boundary
        self.live_plot = live_plot

        # Live plot state
        self._fig = None
        self._ax = None
        self._player_marker = None
        self._boss_marker = None
        self._player_trail = None
        self._boss_trail = None
        self._player_trail_x = []
        self._player_trail_y = []
        self._boss_trail_x = []
        self._boss_trail_y = []

        # Extend observation space
        if hasattr(self.env, 'observation_space') and self.env.observation_space is not None:
            self._extend_obs_space()

    def _extend_obs_space(self):
        """Extend observation space with SDF attributes."""
        if isinstance(self.observation_space, gym.spaces.Dict):
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["sdf_value"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(), dtype=np.float32
            )
            new_spaces["sdf_normal_x"] = gym.spaces.Box(
                low=-1, high=1, shape=(), dtype=np.float32
            )
            new_spaces["sdf_normal_y"] = gym.spaces.Box(
                low=-1, high=1, shape=(), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(new_spaces)

    def _init_live_plot(self):
        """Initialize the live plot."""
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(10, 10))

        # Plot SDF heatmap
        x = np.linspace(self.boundary.x_min, self.boundary.x_max, self.boundary.nx)
        y = np.linspace(self.boundary.y_min, self.boundary.y_max, self.boundary.ny)
        self._ax.contourf(x, y, self.boundary.sdf.T, levels=30, cmap='RdBu_r', alpha=0.5)
        self._ax.contour(x, y, self.boundary.sdf.T, levels=[0], colors='black', linewidths=2)

        # Plot boundary polygon
        poly_x, poly_y = self.boundary.polygon.exterior.xy
        self._ax.plot(poly_x, poly_y, 'k-', linewidth=2, label='Boundary')

        # Initialize markers and trails
        self._player_marker, = self._ax.plot([], [], 'bo', markersize=12, label='Player', zorder=10)
        self._boss_marker, = self._ax.plot([], [], 'ro', markersize=12, label='Boss', zorder=10)
        self._player_trail, = self._ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
        self._boss_trail, = self._ax.plot([], [], 'r-', linewidth=1, alpha=0.5)

        self._ax.set_xlabel('Y (SDF coords)')
        self._ax.set_ylabel('-X (SDF coords)')
        self._ax.set_title('Live Position Tracking with SDF')
        self._ax.legend(loc='upper right')
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)

        plt.show(block=False)
        plt.pause(0.01)

    def _update_live_plot(self, obs):
        """Update the live plot with current positions."""
        if self._fig is None:
            return

        import matplotlib.pyplot as plt

        player_x = obs.get("player_x", 0)
        player_y = obs.get("player_y", 0)
        boss_x = obs.get("boss_x", 0)
        boss_y = obs.get("boss_y", 0)

        # Transform to SDF coords (y, -x)
        player_sdf_x = player_y
        player_sdf_y = -player_x
        boss_sdf_x = boss_y
        boss_sdf_y = -boss_x

        # Update trails
        self._player_trail_x.append(player_sdf_x)
        self._player_trail_y.append(player_sdf_y)
        self._boss_trail_x.append(boss_sdf_x)
        self._boss_trail_y.append(boss_sdf_y)

        # Keep trail length limited
        max_trail = 500
        if len(self._player_trail_x) > max_trail:
            self._player_trail_x = self._player_trail_x[-max_trail:]
            self._player_trail_y = self._player_trail_y[-max_trail:]
            self._boss_trail_x = self._boss_trail_x[-max_trail:]
            self._boss_trail_y = self._boss_trail_y[-max_trail:]

        # Update markers
        self._player_marker.set_data([player_sdf_x], [player_sdf_y])
        self._boss_marker.set_data([boss_sdf_x], [boss_sdf_y])
        self._player_trail.set_data(self._player_trail_x, self._player_trail_y)
        self._boss_trail.set_data(self._boss_trail_x, self._boss_trail_y)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Extend obs space on first reset if needed
        if not hasattr(self, '_obs_space_extended'):
            self._extend_obs_space()
            self._obs_space_extended = True

        # Initialize live plot on first reset
        if self.live_plot and self._fig is None:
            self._init_live_plot()

        # Clear trails on reset
        self._player_trail_x = []
        self._player_trail_y = []
        self._boss_trail_x = []
        self._boss_trail_y = []

        # Query SDF
        obs = self._add_sdf_obs(obs)

        # Update live plot
        if self.live_plot:
            self._update_live_plot(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Query SDF
        obs = self._add_sdf_obs(obs)

        # Update live plot
        if self.live_plot:
            self._update_live_plot(obs)

        return obs, reward, terminated, truncated, info

    def _add_sdf_obs(self, obs):
        """Add SDF observations to obs dict."""
        player_x = obs.get("player_x", 0)
        player_y = obs.get("player_y", 0)

        # Query SDF - boundary uses (y, -x) convention from trace_paths
        # The SDF was built with coords transformed as (global_y, -global_x)
        sdf_x = player_y  # Map player_y to SDF x
        sdf_y = -player_x  # Map -player_x to SDF y

        sdf_value, normal_x, normal_y = self.boundary.query_sdf(sdf_x, sdf_y)

        obs["sdf_value"] = sdf_value
        obs["sdf_normal_x"] = normal_x
        obs["sdf_normal_y"] = normal_y

        return obs

    def close(self):
        """Close the live plot."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
        super().close()


class OOBSafetyWrapper(gym.Wrapper):
    """
    Out-of-bounds detection and recovery via teleportation.

    Uses soft/hard boundary system:
    - Soft boundary: Tracks last safe position when inside
    - Hard boundary: Triggers teleport to last safe position when crossed

    Adds to info:
    - oob_detected: True if player crossed hard boundary
    - teleported: True if teleport was triggered
    - last_safe_xyz: Last known safe position (inside soft boundary)

    Requires player coordinates in obs (player_x, player_y, player_z from EldenGymEnv).

    Args:
        env: EldenGym environment with real coords
        boundary: ArenaBoundary instance with is_inside(x, y) method
        soft_margin: Distance inside the hard boundary for safe zone (default: 3.0)
            - Always positive, represents how far inside the hard boundary
        hard_margin: Distance to extend/shrink hard boundary (default: 0.0)
            - Positive values extend the boundary outward (more permissive)
            - Negative values shrink the boundary inward (more restrictive)
    """

    def __init__(self, env, boundary, soft_margin=3.0, hard_margin=0.0):
        super().__init__(env)
        self.boundary = boundary
        self.soft_margin = soft_margin
        self.hard_margin = hard_margin

        # Last safe GLOBAL position (inside soft boundary)
        self._last_safe_xyz = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize safe position from current GLOBAL position
        player_x = obs.get("player_x", 0)
        player_y = obs.get("player_y", 0)
        player_z = obs.get("player_z", 0)
        self._last_safe_xyz = (player_x, player_y, player_z)

        info["oob_detected"] = False
        info["teleported"] = False
        info["last_safe_xyz"] = self._last_safe_xyz

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get global coords for boundary check
        player_x = obs.get("player_x", 0)
        player_y = obs.get("player_y", 0)
        player_z = obs.get("player_z", 0)

        # Transform to SDF coordinates (y, -x)
        sdf_x = player_y
        sdf_y = -player_x

        # Check boundaries
        sdf_value = self.boundary.nearest_distance(sdf_x, sdf_y)
        inside_hard = sdf_value < self.hard_margin  # Inside hard boundary (extended/shrunk by hard_margin)
        inside_soft = sdf_value < (self.hard_margin - self.soft_margin)  # Inside soft boundary (soft_margin units inside hard)

        oob_detected = not inside_hard
        teleported = False

        if inside_soft:
            # Update safe position with current GLOBAL coords when inside soft boundary
            self._last_safe_xyz = (player_x, player_y, player_z)

        if oob_detected and self._last_safe_xyz is not None:
            # Teleport to last safe GLOBAL position (teleport_to handles conversion)
            safe_x, safe_y, safe_z = self._last_safe_xyz
            self.unwrapped.client.teleport_to(safe_x, safe_y, safe_z)
            teleported = True

        info["oob_detected"] = oob_detected
        info["teleported"] = teleported
        info["last_safe_xyz"] = self._last_safe_xyz
        info["sdf_value"] = sdf_value
        info["inside_hard"] = inside_hard
        info["inside_soft"] = inside_soft

        return obs, reward, terminated, truncated, info


class DodgePolicyRewardWrapper(gym.Wrapper):
    """
    Reward shaping for dodge policy training.

    Applies configurable penalties for:
    - Taking damage (hit by boss)
    - Dodging (to prevent spam)
    - Being in danger zone (between soft and hard boundary)
    - Crossing hard boundary (OOB/teleported)

    Requires HPRefundWrapper and OOBSafetyWrapper to be applied first.

    Args:
        env: EldenGym environment with HPRefundWrapper and OOBSafetyWrapper
        dodge_action_idx: Index of dodge action in action space
        hit_penalty: Penalty for taking damage (default: -1.0)
        dodge_penalty: Penalty per dodge to prevent spam (default: -0.01)
        danger_zone_penalty: Penalty for being between soft/hard boundary (default: -0.1)
        oob_penalty: Penalty for crossing hard boundary/teleport (default: -1.0)
    """

    def __init__(
        self,
        env,
        dodge_action_idx: int,
        hit_penalty: float = -1.0,
        dodge_penalty: float = -0.01,
        danger_zone_penalty: float = -0.1,
        oob_penalty: float = -1.0,
    ):
        super().__init__(env)
        self.dodge_action_idx = dodge_action_idx
        self.hit_penalty = hit_penalty
        self.dodge_penalty = dodge_penalty
        self.danger_zone_penalty = danger_zone_penalty
        self.oob_penalty = oob_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Start with base reward (usually 0 or from underlying reward function)
        shaped_reward = reward

        # Penalty for taking damage
        damage_taken = info.get("player_damage_taken", 0)
        if damage_taken > 0:
            shaped_reward += self.hit_penalty
            info["reward_hit_penalty"] = self.hit_penalty

        # Penalty for dodging (spam prevention)
        if action[self.dodge_action_idx] == 1:
            shaped_reward += self.dodge_penalty
            info["reward_dodge_penalty"] = self.dodge_penalty

        # Penalty for danger zone (between soft and hard boundary)
        inside_hard = info.get("inside_hard", True)
        inside_soft = info.get("inside_soft", True)
        if inside_hard and not inside_soft:
            shaped_reward += self.danger_zone_penalty
            info["reward_danger_zone_penalty"] = self.danger_zone_penalty

        # Penalty for OOB (crossed hard boundary, teleported)
        teleported = info.get("teleported", False)
        if teleported:
            shaped_reward += self.oob_penalty
            info["reward_oob_penalty"] = self.oob_penalty

        info["reward_raw"] = reward
        info["reward_shaped"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info
