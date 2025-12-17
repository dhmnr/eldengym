from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for custom reward functions"""

    @abstractmethod
    def calculate(self, obs, info, prev_info=None):
        """
        Calculate reward based on observation and info.

        Args:
            obs: dict, current observation (frame + memory attributes)
            info: dict, processed info from observation
            prev_info: dict, previous info (None on first step)

        Returns:
            float: reward value
        """
        pass

    @abstractmethod
    def is_done(self, obs, info):
        """
        Determine if episode should terminate.

        Args:
            obs: dict, current observation
            info: dict, processed info

        Returns:
            bool: whether episode should terminate
        """
        pass


class ScoreDeltaReward(RewardFunction):
    """
    Reward based on change in a score metric.

    Args:
        score_key: str, key in info dict to use as score (default: "player_hp_normalized")
        win_bonus: float, bonus reward for winning (boss defeated) (default: 100.0)
        loss_penalty: float, penalty for losing (player defeated) (default: -100.0)
    """

    def __init__(
        self, score_key="player_hp_normalized", win_bonus=100.0, loss_penalty=-100.0
    ):
        self.score_key = score_key
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty

    def calculate(self, obs, info, prev_info=None):
        if prev_info is None:
            return 0.0

        # Calculate score delta
        current_score = info.get(self.score_key, 0.0)
        prev_score = prev_info.get(self.score_key, 0.0)
        reward = current_score - prev_score

        # Add win/loss bonuses
        if info.get("boss_hp_normalized", 1.0) <= 0:
            reward += self.win_bonus
        elif info.get("player_hp_normalized", 1.0) <= 0:
            reward += self.loss_penalty

        return reward

    def is_done(self, obs, info):
        # Episode ends when player or boss dies
        player_hp = info.get("player_hp_normalized", 1.0)
        boss_hp = info.get("boss_hp_normalized", 1.0)
        return player_hp <= 0 or boss_hp <= 0


class BossDefeatReward(RewardFunction):
    """
    Sparse reward - only rewards for boss defeat, penalizes for death.

    Args:
        boss_defeat_reward: float, reward for defeating boss (default: 1.0)
        player_death_penalty: float, penalty for player death (default: -1.0)
        time_penalty: float, small penalty per step to encourage efficiency (default: -0.001)
    """

    def __init__(
        self, boss_defeat_reward=1.0, player_death_penalty=-1.0, time_penalty=-0.001
    ):
        self.boss_defeat_reward = boss_defeat_reward
        self.player_death_penalty = player_death_penalty
        self.time_penalty = time_penalty

    def calculate(self, obs, info, prev_info=None):
        reward = self.time_penalty

        # Check for episode ending events
        if info.get("boss_hp_normalized", 1.0) <= 0:
            reward += self.boss_defeat_reward
        elif info.get("player_hp_normalized", 1.0) <= 0:
            reward += self.player_death_penalty

        return reward

    def is_done(self, obs, info):
        player_hp = info.get("player_hp_normalized", 1.0)
        boss_hp = info.get("boss_hp_normalized", 1.0)
        return player_hp <= 0 or boss_hp <= 0


class CustomReward(RewardFunction):
    """
    Customizable reward with multiple components.

    Args:
        hp_delta_weight: float, weight for player HP changes (default: 1.0)
        boss_damage_weight: float, weight for damage dealt to boss (default: 0.5)
        survival_bonus: float, small bonus per step for staying alive (default: 0.01)
        win_bonus: float, bonus for defeating boss (default: 100.0)
        loss_penalty: float, penalty for player death (default: -100.0)
    """

    def __init__(
        self,
        hp_delta_weight=1.0,
        boss_damage_weight=0.5,
        survival_bonus=0.01,
        win_bonus=100.0,
        loss_penalty=-100.0,
    ):
        self.hp_delta_weight = hp_delta_weight
        self.boss_damage_weight = boss_damage_weight
        self.survival_bonus = survival_bonus
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty

    def calculate(self, obs, info, prev_info=None):
        if prev_info is None:
            return 0.0

        reward = 0.0

        # Player HP delta (positive is good)
        player_hp = info.get("player_hp_normalized", 0.0)
        prev_player_hp = prev_info.get("player_hp_normalized", 0.0)
        hp_delta = player_hp - prev_player_hp
        reward += self.hp_delta_weight * hp_delta

        # Boss HP delta (negative is good - we want to damage boss)
        boss_hp = info.get("boss_hp_normalized", 1.0)
        prev_boss_hp = prev_info.get("boss_hp_normalized", 1.0)
        boss_delta = prev_boss_hp - boss_hp  # Flipped: damage is positive
        reward += self.boss_damage_weight * boss_delta

        # Survival bonus
        if player_hp > 0:
            reward += self.survival_bonus

        # Win/loss bonuses
        if boss_hp <= 0:
            reward += self.win_bonus
        elif player_hp <= 0:
            reward += self.loss_penalty

        return reward

    def is_done(self, obs, info):
        player_hp = info.get("player_hp_normalized", 1.0)
        boss_hp = info.get("boss_hp_normalized", 1.0)
        return player_hp <= 0 or boss_hp <= 0
