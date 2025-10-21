"""
EldenGym - A Gymnasium environment for Elden Ring using Siphon memory reading
"""

from .client import SiphonClient, EldenClient
from .env import EldenGymEnv
from .rewards import RewardFunction, ScoreDeltaReward, CustomReward
from .wrappers import FrameStack, ResizeFrame, GrayscaleFrame

__version__ = "0.1.0"
__all__ = [
    "SiphonClient",
    "EldenClient",
    "EldenGymEnv",
    "RewardFunction",
    "ScoreDeltaReward",
    "CustomReward",
    "FrameStack",
    "ResizeFrame",
    "GrayscaleFrame",
]
