"""
Random Dodge + Movement Policy with Full Wrapper Stack

Demonstrates:
1. Real coordinates (player_xyz, boss_xyz, dist_to_boss, boss_z_relative)
2. Animation tracking (boss_anim_id, elapsed_frames)
3. SDF observations (sdf_value, sdf_normal_x/y)
4. OOB safety (teleport on boundary crossing)
5. HP refund (infinite health for data collection)
6. Live plot of positions with SDF visualization

Usage:
    python examples/random_dodge_movement.py --boundary path/to/arena_boundary.json
"""

import time
import argparse
import numpy as np

import eldengym
from eldengym import (
    ArenaBoundary,
    AnimFrameWrapper,
    SDFObsWrapper,
    OOBSafetyWrapper,
    HPRefundWrapper,
)


def run_random_dodge_movement(
    boundary_path: str,
    num_steps: int = 1000,
    host: str = "192.168.48.1:50051",
    soft_margin: float = 5.0,
    hard_margin: float = 5.0,
    live_plot: bool = True,
    launch_game: bool = False,
):
    """
    Run random dodge + movement policy with full wrapper stack.

    Only activates movement (WASD) and dodge (SPACE) actions randomly.
    """
    # Load arena boundary
    print(f"Loading arena boundary from {boundary_path}...")
    boundary = ArenaBoundary.load(boundary_path)
    print(f"  Bounds: x=[{boundary.x_min:.1f}, {boundary.x_max:.1f}], y=[{boundary.y_min:.1f}, {boundary.y_max:.1f}]")

    # Create environment
    print("Creating environment...")
    env = eldengym.make("Margit-v0", launch_game=launch_game, host=host)

    # Apply wrappers (order matters!)
    print("Applying wrappers...")
    env = HPRefundWrapper(env, refund_player=True, refund_boss=False)
    env = AnimFrameWrapper(env)
    env = SDFObsWrapper(env, boundary=boundary, live_plot=live_plot)
    env = OOBSafetyWrapper(env, boundary=boundary, soft_margin=soft_margin, hard_margin=hard_margin)

    print("=" * 70)
    print("Random Dodge + Movement Policy")
    print("=" * 70)
    print(f"Steps: {num_steps}")
    print(f"Soft margin: {soft_margin} (soft threshold: sdf < {hard_margin - soft_margin})")
    print(f"Hard margin: {hard_margin} (hard threshold: sdf < {hard_margin})")
    print(f"Live plot: {live_plot}")
    print(f"Action space: {env.action_space}")
    print("=" * 70)

    # Get action indices for movement and dodge
    # Typical keybind order: move_forward(0), move_backward(1), move_left(2), move_right(3), dodge(4), ...
    action_keys = env.action_keys if hasattr(env, 'action_keys') else []
    print(f"Action keys: {action_keys}")

    # Find movement and dodge indices
    movement_actions = []
    dodge_action = None
    for i, key in enumerate(action_keys):
        if 'move' in key.lower() or key.lower() in ['w', 'a', 's', 'd']:
            movement_actions.append(i)
        if 'dodge' in key.lower() or 'roll' in key.lower() or key.lower() == 'space':
            dodge_action = i

    print(f"Movement action indices: {movement_actions}")
    print(f"Dodge action index: {dodge_action}")
    print("=" * 70)

    # Reset environment
    obs, info = env.reset()

    print("\nObservation keys:", list(obs.keys()))
    print("-" * 70)

    # Stats tracking
    total_damage = 0
    oob_count = 0
    teleport_count = 0

    for step in range(num_steps):
        # Random dodge + movement action
        action = np.zeros(env.action_space.n, dtype=np.int8)

        # Random movement (pick 0-2 movement keys)
        if movement_actions:
            num_moves = np.random.randint(0, 3)
            chosen_moves = np.random.choice(movement_actions, size=min(num_moves, len(movement_actions)), replace=False)
            for idx in chosen_moves:
                action[idx] = 1

        # Random dodge (20% chance)
        if dodge_action is not None and np.random.random() < 0.2:
            action[dodge_action] = 1

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract observations for dodge policy
        boss_anim_id = obs.get("boss_anim_id", 0)
        elapsed_frames = obs.get("elapsed_frames", 0)
        dist_to_boss = obs.get("dist_to_boss", 0)
        boss_z_relative = obs.get("boss_z_relative", 0)
        sdf_value = obs.get("sdf_value", 0)
        sdf_normal_x = obs.get("sdf_normal_x", 0)
        sdf_normal_y = obs.get("sdf_normal_y", 0)

        # Info
        player_damage = info.get("player_damage_taken", 0)
        oob_detected = info.get("oob_detected", False)
        teleported = info.get("teleported", False)

        total_damage += player_damage
        if oob_detected:
            oob_count += 1
        if teleported:
            teleport_count += 1

        # Print state every 10 steps or on events
        inside_hard = info.get("inside_hard", False)
        inside_soft = info.get("inside_soft", False)
        if step % 10 == 0 or player_damage > 0 or oob_detected:
            hard_str = "H" if inside_hard else "h"
            soft_str = "S" if inside_soft else "s"
            print(
                f"Step {step:4d} | "
                f"AnimID: {int(boss_anim_id):8d} | "
                f"Frames: {int(elapsed_frames):3d} | "
                f"Dist: {dist_to_boss:5.1f} | "
                f"Z: {boss_z_relative:+5.2f} | "
                f"SDF: {sdf_value:+6.2f} [{hard_str}{soft_str}] | "
                f"Norm: ({sdf_normal_x:+.2f}, {sdf_normal_y:+.2f})"
            )

            if player_damage > 0:
                print(f"        >>> DAMAGE: {player_damage:.0f}")
            if oob_detected:
                print(f"        >>> OOB DETECTED! Teleported: {teleported}")

        if terminated or truncated:
            print("\n" + "=" * 70)
            if info.get("boss_hp_normalized", 1.0) <= 0:
                print("BOSS DEFEATED!")
            else:
                print("Episode ended")
            print("=" * 70)
            break

        # Small delay for visualization
        time.sleep(0.03)

    # Final stats
    print("\n" + "=" * 70)
    print("Session Statistics")
    print("=" * 70)
    print(f"Total steps: {step + 1}")
    print(f"Total damage taken: {total_damage:.0f}")
    print(f"OOB detections: {oob_count}")
    print(f"Teleports: {teleport_count}")
    print("=" * 70)

    env.close()
    print("\nEnvironment closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Random dodge + movement policy with full wrapper stack"
    )
    parser.add_argument(
        "--boundary",
        default="/home/dm/ProjectRanni/paths/arena_boundary.json",
        help="Path to arena boundary JSON file",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to run"
    )
    parser.add_argument(
        "--host", default="192.168.48.1:50051", help="Siphon server address"
    )
    parser.add_argument(
        "--soft-margin", type=float, default=5.0, help="Distance inside hard boundary for safe zone"
    )
    parser.add_argument(
        "--hard-margin", type=float, default=0.0, help="Hard boundary margin (+extends, -shrinks)"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable live plot"
    )
    parser.add_argument(
        "--launch-game", action="store_true", help="Launch game"
    )

    args = parser.parse_args()

    run_random_dodge_movement(
        boundary_path=args.boundary,
        num_steps=args.steps,
        host=args.host,
        soft_margin=args.soft_margin,
        hard_margin=args.hard_margin,
        live_plot=not args.no_plot,
        launch_game=args.launch_game,
    )


if __name__ == "__main__":
    main()
