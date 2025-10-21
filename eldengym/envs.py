"""
Register all default EldenGym environments.

This module registers pre-configured environments for common boss fights.
"""

from .env import EldenGymEnv
from .registry import register

# Register Margit environments
register(
    id="EldenRing-Margit-v0",
    entry_point=EldenGymEnv,
    kwargs={
        "scenario_name": "margit",
        "action_mode": "discrete",
        "frame_skip": 4,
        "game_speed": 1.0,
    },
)

register(
    id="EldenRing-Margit-Easy-v0",
    entry_point=EldenGymEnv,
    kwargs={
        "scenario_name": "margit",
        "action_mode": "discrete",
        "frame_skip": 4,
        "game_speed": 0.5,  # Slower game speed
    },
)

register(
    id="EldenRing-Margit-MultiBinary-v0",
    entry_point=EldenGymEnv,
    kwargs={
        "scenario_name": "margit",
        "action_mode": "multi_binary",
        "frame_skip": 4,
        "game_speed": 1.0,
    },
)

# # Register Godrick environments
# register(
#     id='EldenRing-Godrick-v0',
#     entry_point=EldenGymEnv,
#     kwargs={
#         'scenario_name': 'godrick',
#         'action_mode': 'discrete',
#         'frame_skip': 4,
#         'game_speed': 1.0,
#     },
# )

# register(
#     id='EldenRing-Godrick-Easy-v0',
#     entry_point=EldenGymEnv,
#     kwargs={
#         'scenario_name': 'godrick',
#         'action_mode': 'discrete',
#         'frame_skip': 4,
#         'game_speed': 0.5,
#     },
# )

# register(
#     id='EldenRing-Godrick-MultiBinary-v0',
#     entry_point=EldenGymEnv,
#     kwargs={
#         'scenario_name': 'godrick',
#         'action_mode': 'multi_binary',
#         'frame_skip': 4,
#         'game_speed': 1.0,
#     },
# )

# # Generic environment that can be customized
# register(
#     id='EldenRing-v0',
#     entry_point=EldenGymEnv,
#     kwargs={
#         'action_mode': 'discrete',
#         'frame_skip': 4,
#         'game_speed': 1.0,
#     },
# )
