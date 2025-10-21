import gymnasium as gym
import numpy as np
from collections import deque


class FrameStack(gym.ObservationWrapper):
    """Stack last N frames"""

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
    """Resize frames to target shape"""

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
    """Convert to grayscale"""

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
