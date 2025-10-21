from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for custom reward functions"""

    @abstractmethod
    def calculate(self, obs, info, prev_info=None):
        """
        Args:
            obs: current observation (frame)
            info: dict with current memory values
            prev_info: dict with previous memory values (None on first step)

        Returns:
            float: reward value
        """
        pass

    @abstractmethod
    def is_done(self, obs, info):
        """
        Args:
            obs: current observation
            info: dict with current memory values

        Returns:
            bool: whether episode should terminate
        """
        pass


class ScoreDeltaReward(RewardFunction):
    """Example: reward based on score increase"""

    def __init__(self, score_key="player_hp"):
        self.score_key = score_key

    def calculate(self, obs, info, prev_info=None):
        if prev_info is None:
            return 0.0
        return info[self.score_key] - prev_info[self.score_key]

    def is_done(self, obs, info):
        return info.get("player_hp", 1) <= 100


class CustomReward(RewardFunction):
    """User can subclass this for game-specific logic"""

    def __init__(self, score_weight=1.0, hp_weight=0.1):
        self.score_weight = score_weight
        self.hp_weight = hp_weight

    def calculate(self, obs, info, prev_info=None):
        if prev_info is None:
            return 0.0

        score_delta = info.get("score", 0) - prev_info.get("score", 0)
        hp_delta = info.get("hp", 0) - prev_info.get("hp", 0)

        return self.score_weight * score_delta + self.hp_weight * hp_delta

    def is_done(self, obs, info):
        return info.get("hp", 100) <= 0
