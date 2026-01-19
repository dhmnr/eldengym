"""
HP Refund Wrapper Example

Demonstrates using HPRefundWrapper for data collection where you want to:
1. Prevent episode termination from player death
2. Track damage taken for reward shaping
3. Run indefinitely for exploration/data collection

Usage:
    python examples/hp_refund_wrapper.py
"""

import time
import argparse
import eldengym
from eldengym import HPRefundWrapper
import numpy as np


def run_policy(
    num_steps: int = 1000,
    host: str = "192.168.48.1:50051",
    damage_penalty_scale: float = 10.0,
    launch_game: bool = False,
):
    """
    Run random policy with HP refund wrapper.

    Args:
        num_steps: Number of steps to run
        host: Siphon server address
        damage_penalty_scale: Multiplier for damage penalty in shaped reward
        launch_game: Whether to launch the game (False if already running)
    """
    # Create base environment
    env = eldengym.make(
        "Margit-v0",
        launch_game=launch_game,
        host=host,
    )

    # Wrap with HP refund - player HP is refunded after each step
    env = HPRefundWrapper(env, refund_player=True, refund_boss=False)

    print("=" * 60)
    print("Random Policy with HP Refund")
    print("=" * 60)
    print(f"Steps: {num_steps}")
    print(f"Damage penalty scale: {damage_penalty_scale}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)

    observation, info = env.reset()

    print(f"\nObservation keys: {list(observation.keys())}")
    print(f"Frame shape: {observation['frame'].shape}")
    print("-" * 60)

    total_damage = 0
    total_shaped_reward = 0
    damage_events = 0

    for step in range(num_steps):
        # Sample random action
        action = np.zeros(13)

        observation, reward, terminated, truncated, info = env.step(action)

        # Get damage taken this step (added by HPRefundWrapper)
        damage_taken = info.get("player_damage_taken", 0)
        damage_normalized = info.get("player_damage_taken_normalized", 0)

        # Calculate shaped reward with damage penalty
        damage_penalty = -damage_normalized * damage_penalty_scale
        shaped_reward = reward + damage_penalty

        # Track statistics
        total_damage += damage_taken
        total_shaped_reward += shaped_reward
        if damage_taken > 0:
            damage_events += 1

        # Display progress
        player_hp_pct = info.get("player_hp_normalized", 0) * 100
        boss_hp_pct = info.get("boss_hp_normalized", 1) * 100

        # Show damage events prominently
        if damage_taken > 0:
            print(
                f"Step {step:4d} | "
                f"Player HP: {player_hp_pct:5.1f}% | "
                f"Boss HP: {boss_hp_pct:5.1f}% | "
                f"DMG: {damage_taken:4.0f} | "
                f"Penalty: {damage_penalty:+.3f} | "
                f"Shaped R: {shaped_reward:+.3f}"
            )
        elif step % 100 == 0:
            # Periodic status update
            print(
                f"Step {step:4d} | "
                f"Player HP: {player_hp_pct:5.1f}% | "
                f"Boss HP: {boss_hp_pct:5.1f}% | "
                f"Total DMG: {total_damage:.0f}"
            )

        # Note: With HP refund, terminated should rarely be True
        # unless boss is defeated or other termination condition
        if terminated:
            if info.get("boss_hp_normalized", 1.0) <= 0:
                print("\n" + "=" * 60)
                print("BOSS DEFEATED!")
                print("=" * 60)
            break

        # Small delay for readability (remove for actual training)
        time.sleep(0.05)

    # Final statistics
    print("\n" + "=" * 60)
    print("Session Statistics")
    print("=" * 60)
    print(f"Total steps: {step + 1}")
    print(f"Total damage taken: {total_damage:.0f}")
    print(f"Damage events: {damage_events}")
    print(f"Avg damage per event: {total_damage / max(damage_events, 1):.1f}")
    print(f"Total shaped reward: {total_shaped_reward:.3f}")
    print("=" * 60)

    env.close()
    print("\nEnvironment closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Random policy with HP refund for data collection"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to run"
    )
    parser.add_argument(
        "--host", default="192.168.48.1:50051", help="Siphon server address"
    )
    parser.add_argument(
        "--damage-penalty",
        type=float,
        default=10.0,
        help="Scale for damage penalty in shaped reward",
    )
    parser.add_argument(
        "--launch-game",
        action="store_true",
        help="Launch game (default: assume already running)",
    )

    args = parser.parse_args()

    run_policy(
        num_steps=args.steps,
        host=args.host,
        damage_penalty_scale=args.damage_penalty,
        launch_game=args.launch_game,
    )


if __name__ == "__main__":
    main()
