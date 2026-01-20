"""
Random Dodge + Movement Policy with Full Wrapper Stack

Demonstrates:
1. Filtered action space (movement + dodge only)
2. Real coordinates (player_xyz, boss_xyz, dist_to_boss, boss_z_relative)
3. Animation tracking (boss_anim_id, elapsed_frames)
4. SDF observations (sdf_value, sdf_normal_x/y)
5. OOB safety (teleport on boundary crossing)
6. HP refund (infinite health for data collection)
7. Reward shaping (hit, dodge, danger zone, OOB penalties)
8. Live plot of positions with SDF visualization

Usage:
    python examples/random_dodge_movement.py --boundary path/to/arena_boundary.json
"""

import time
import argparse
import numpy as np
from collections import deque

import eldengym
from eldengym import (
    ArenaBoundary,
    AnimFrameWrapper,
    SDFObsWrapper,
    OOBSafetyWrapper,
    HPRefundWrapper,
    DodgePolicyRewardWrapper,
)


# Actions for dodge policy
DODGE_POLICY_ACTIONS = [
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "dodge_roll/dash",
]

# Action indices (based on order in DODGE_POLICY_ACTIONS)
ACTION_FORWARD = 0
ACTION_BACK = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_DODGE = 4


def format_action(action):
    """Format action array as readable string."""
    parts = []
    if action[ACTION_FORWARD]:
        parts.append("F")
    if action[ACTION_BACK]:
        parts.append("B")
    if action[ACTION_LEFT]:
        parts.append("L")
    if action[ACTION_RIGHT]:
        parts.append("R")
    if action[ACTION_DODGE]:
        parts.append("DODGE")
    return "+".join(parts) if parts else "NONE"


def run_random_dodge_movement(
    boundary_path: str,
    num_steps: int = 1000,
    host: str = "192.168.48.1:50051",
    soft_margin: float = 5.0,
    hard_margin: float = 0.0,
    hit_penalty: float = -1.0,
    dodge_penalty: float = -0.01,
    danger_zone_penalty: float = -0.1,
    oob_penalty: float = -1.0,
    dodge_prob: float = 0.2,
    live_plot: bool = True,
    launch_game: bool = False,
    log_interval: int = 10,
):
    """
    Run random dodge + movement policy with full wrapper stack.

    Args:
        boundary_path: Path to arena boundary JSON file
        num_steps: Number of steps to run
        host: Siphon server address
        soft_margin: Distance inside hard boundary for safe zone
        hard_margin: Hard boundary margin (+extends, -shrinks)
        hit_penalty: Reward penalty for taking damage
        dodge_penalty: Reward penalty per dodge (spam prevention)
        danger_zone_penalty: Reward penalty for being between soft/hard boundary
        oob_penalty: Reward penalty for crossing hard boundary
        dodge_prob: Probability of dodging each step
        live_plot: Enable live visualization
        launch_game: Whether to launch the game
        log_interval: Steps between log outputs
    """
    # Load arena boundary
    print("=" * 80)
    print("DODGE POLICY - Random Movement Demo")
    print("=" * 80)
    print(f"\nLoading arena boundary from {boundary_path}...")
    boundary = ArenaBoundary.load(boundary_path)
    print(f"  Bounds: x=[{boundary.x_min:.1f}, {boundary.x_max:.1f}], "
          f"y=[{boundary.y_min:.1f}, {boundary.y_max:.1f}]")

    # Create environment with filtered action space
    print(f"\nCreating environment with filtered action space...")
    print(f"  Actions: {DODGE_POLICY_ACTIONS}")
    env = eldengym.make(
        "Margit-v0",
        launch_game=launch_game,
        host=host,
        actions=DODGE_POLICY_ACTIONS,
    )
    print(f"  Action space: {env.action_space}")
    print(f"  Action keys: {env.action_keys}")

    # Apply wrappers (order matters!)
    print(f"\nApplying wrappers...")
    env = HPRefundWrapper(env, refund_player=True, refund_boss=False)
    print(f"  [1] HPRefundWrapper (refund_player=True)")
    env = AnimFrameWrapper(env)
    print(f"  [2] AnimFrameWrapper")
    env = SDFObsWrapper(env, boundary=boundary, live_plot=live_plot)
    print(f"  [3] SDFObsWrapper (live_plot={live_plot})")
    env = OOBSafetyWrapper(env, boundary=boundary, soft_margin=soft_margin, hard_margin=hard_margin)
    print(f"  [4] OOBSafetyWrapper (soft={soft_margin}, hard={hard_margin})")
    print(f"      Soft threshold: sdf < {hard_margin - soft_margin:.1f}")
    print(f"      Hard threshold: sdf < {hard_margin:.1f}")
    env = DodgePolicyRewardWrapper(
        env,
        dodge_action_idx=ACTION_DODGE,
        hit_penalty=hit_penalty,
        dodge_penalty=dodge_penalty,
        danger_zone_penalty=danger_zone_penalty,
        oob_penalty=oob_penalty,
    )
    print(f"  [5] DodgePolicyRewardWrapper")
    print(f"      hit_penalty={hit_penalty}, dodge_penalty={dodge_penalty}")
    print(f"      danger_zone_penalty={danger_zone_penalty}, oob_penalty={oob_penalty}")

    # Reset environment
    print(f"\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation keys: {list(obs.keys())}")

    # Stats tracking
    stats = {
        "total_reward": 0.0,
        "total_damage": 0.0,
        "hit_count": 0,
        "dodge_count": 0,
        "danger_zone_steps": 0,
        "oob_count": 0,
        "teleport_count": 0,
    }

    # Rolling averages (last 100 steps)
    reward_history = deque(maxlen=100)
    damage_history = deque(maxlen=100)

    # Animation tracking
    last_anim_id = None
    anim_changes = 0

    print("\n" + "=" * 80)
    print("RUNNING POLICY")
    print("=" * 80)
    print(f"{'Step':>6} | {'Action':<12} | {'Reward':>7} | {'Dist':>5} | "
          f"{'SDF':>6} | {'Zone':<4} | {'Anim':>10} | {'Frames':>3} | Event")
    print("-" * 80)

    start_time = time.time()

    for step in range(num_steps):
        # Random dodge + movement action
        action = np.zeros(env.action_space.n, dtype=np.int8)

        # Random movement (pick 0-2 movement directions)
        num_moves = np.random.randint(0, 3)
        if num_moves > 0:
            move_indices = [ACTION_FORWARD, ACTION_BACK, ACTION_LEFT, ACTION_RIGHT]
            chosen_moves = np.random.choice(move_indices, size=num_moves, replace=False)
            for idx in chosen_moves:
                action[idx] = 1

        # Random dodge
        if np.random.random() < dodge_prob:
            action[ACTION_DODGE] = 1

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Update stats
        stats["total_reward"] += reward
        reward_history.append(reward)

        damage = info.get("player_damage_taken", 0)
        if damage > 0:
            stats["total_damage"] += damage
            stats["hit_count"] += 1
        damage_history.append(damage)

        if action[ACTION_DODGE]:
            stats["dodge_count"] += 1

        inside_hard = info.get("inside_hard", True)
        inside_soft = info.get("inside_soft", True)
        if inside_hard and not inside_soft:
            stats["danger_zone_steps"] += 1

        if info.get("oob_detected", False):
            stats["oob_count"] += 1
        if info.get("teleported", False):
            stats["teleport_count"] += 1

        # Animation tracking
        boss_anim_id = obs.get("boss_anim_id", 0)
        if last_anim_id is not None and boss_anim_id != last_anim_id:
            anim_changes += 1
        last_anim_id = boss_anim_id

        # Extract obs values
        dist_to_boss = obs.get("dist_to_boss", 0)
        sdf_value = info.get("sdf_value", 0)
        elapsed_frames = obs.get("elapsed_frames", 0)

        # Zone indicator
        if inside_soft:
            zone = "SAFE"
        elif inside_hard:
            zone = "WARN"
        else:
            zone = "OOB!"

        # Build event string
        events = []
        if damage > 0:
            events.append(f"HIT({damage:.0f})")
        if info.get("teleported", False):
            events.append("TELEPORT")
        if "reward_danger_zone_penalty" in info:
            events.append("DANGER")
        event_str = " ".join(events)

        # Log output
        should_log = (
            step % log_interval == 0 or
            damage > 0 or
            info.get("teleported", False) or
            terminated or truncated
        )

        if should_log:
            print(
                f"{step:6d} | {format_action(action):<12} | {reward:+7.3f} | "
                f"{dist_to_boss:5.1f} | {sdf_value:+6.2f} | {zone:<4} | "
                f"{int(boss_anim_id):>10d} | {int(elapsed_frames):>3d} | {event_str}"
            )

        if terminated or truncated:
            print("\n" + "=" * 80)
            if info.get("boss_hp_normalized", 1.0) <= 0:
                print("BOSS DEFEATED!")
            else:
                print("EPISODE ENDED")
            break

        # Small delay for visualization
        time.sleep(0.03)

    elapsed_time = time.time() - start_time
    steps_completed = step + 1

    # Final stats
    print("\n" + "=" * 80)
    print("SESSION STATISTICS")
    print("=" * 80)
    print(f"\nPerformance:")
    print(f"  Steps completed:    {steps_completed}")
    print(f"  Elapsed time:       {elapsed_time:.1f}s")
    print(f"  Steps/second:       {steps_completed / elapsed_time:.1f}")

    print(f"\nRewards:")
    print(f"  Total reward:       {stats['total_reward']:+.3f}")
    print(f"  Avg reward/step:    {stats['total_reward'] / steps_completed:+.5f}")
    if reward_history:
        print(f"  Avg reward (last {len(reward_history)}): {np.mean(reward_history):+.5f}")

    print(f"\nDamage:")
    print(f"  Total damage:       {stats['total_damage']:.0f}")
    print(f"  Hit count:          {stats['hit_count']}")
    print(f"  Hit rate:           {stats['hit_count'] / steps_completed * 100:.2f}%")

    print(f"\nActions:")
    print(f"  Dodge count:        {stats['dodge_count']}")
    print(f"  Dodge rate:         {stats['dodge_count'] / steps_completed * 100:.1f}%")

    print(f"\nBoundary:")
    print(f"  Danger zone steps:  {stats['danger_zone_steps']}")
    print(f"  Danger zone rate:   {stats['danger_zone_steps'] / steps_completed * 100:.1f}%")
    print(f"  OOB detections:     {stats['oob_count']}")
    print(f"  Teleports:          {stats['teleport_count']}")

    print(f"\nAnimation:")
    print(f"  Animation changes:  {anim_changes}")
    print(f"  Avg change rate:    {anim_changes / steps_completed * 100:.1f}%")

    print("=" * 80)

    env.close()
    print("\nEnvironment closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Random dodge + movement policy with full wrapper stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument(
        "--boundary",
        default="/home/dm/ProjectRanni/paths/arena_boundary.json",
        help="Path to arena boundary JSON file",
    )
    parser.add_argument("--host", default="192.168.48.1:50051", help="Siphon server address")
    parser.add_argument("--launch-game", action="store_true", help="Launch game")

    # Episode
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log outputs")

    # Boundary margins
    parser.add_argument("--soft-margin", type=float, default=5.0, help="Soft boundary margin")
    parser.add_argument("--hard-margin", type=float, default=0.0, help="Hard boundary margin")

    # Reward penalties
    parser.add_argument("--hit-penalty", type=float, default=-1.0, help="Penalty for getting hit")
    parser.add_argument("--dodge-penalty", type=float, default=-0.01, help="Penalty per dodge")
    parser.add_argument("--danger-penalty", type=float, default=-0.1, help="Penalty for danger zone")
    parser.add_argument("--oob-penalty", type=float, default=-1.0, help="Penalty for OOB/teleport")

    # Policy
    parser.add_argument("--dodge-prob", type=float, default=0.2, help="Probability of dodging")

    # Visualization
    parser.add_argument("--no-plot", action="store_true", help="Disable live plot")

    args = parser.parse_args()

    run_random_dodge_movement(
        boundary_path=args.boundary,
        num_steps=args.steps,
        host=args.host,
        soft_margin=args.soft_margin,
        hard_margin=args.hard_margin,
        hit_penalty=args.hit_penalty,
        dodge_penalty=args.dodge_penalty,
        danger_zone_penalty=args.danger_penalty,
        oob_penalty=args.oob_penalty,
        dodge_prob=args.dodge_prob,
        live_plot=not args.no_plot,
        launch_game=args.launch_game,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
