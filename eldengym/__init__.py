"""
EldenGym - A Gymnasium environment for Elden Ring using Siphon memory reading
"""

from .client import SiphonClient, EldenClient
from .env import EldenGymEnv
from .rewards import (
    RewardFunction,
    ScoreDeltaReward,
    BossDefeatReward,
    CustomReward,
)
from .wrappers import (
    FrameStack,
    ResizeFrame,
    GrayscaleFrame,
    DictFrameStack,
    DictResizeFrame,
    DictGrayscaleFrame,
    NormalizeMemoryAttributes,
    HPRefundWrapper,
    AnimFrameWrapper,
    SDFObsWrapper,
    OOBSafetyWrapper,
    DodgePolicyRewardWrapper,
)
from .arena_boundary import ArenaBoundary, BoundaryDistances
from .registry import make, register, list_envs

# Import envs module to trigger environment registrations
from . import envs

__version__ = "0.1.0"
__all__ = [
    "SiphonClient",
    "EldenClient",
    "EldenGymEnv",
    "RewardFunction",
    "ScoreDeltaReward",
    "BossDefeatReward",
    "CustomReward",
    "FrameStack",
    "ResizeFrame",
    "GrayscaleFrame",
    "DictFrameStack",
    "DictResizeFrame",
    "DictGrayscaleFrame",
    "NormalizeMemoryAttributes",
    "HPRefundWrapper",
    "AnimFrameWrapper",
    "SDFObsWrapper",
    "OOBSafetyWrapper",
    "DodgePolicyRewardWrapper",
    "ArenaBoundary",
    "BoundaryDistances",
    "make",
    "register",
    "list_envs",
    "envs",
]
