#!/usr/bin/env python3
"""
Advanced timing example for Elden Ring environment
Demonstrates animation phases, stepping strategies, and observation sampling
"""

import time
from eldengym.env import EldenGymEnv

def demonstrate_animation_phases():
    """Demonstrate animation phase tracking"""
    print("=== Animation Phase Tracking ===")
    
    env = EldenGymEnv(
        stepping_strategy='adaptive',
        observation_sampling='smart',
        agent_frequency=5.0  # 5 Hz agent frequency
    )
    
    obs, info = env.reset()
    print(f"Initial state: {obs['action_state']}")
    
    # Execute an attack (has startup, active, recovery phases)
    print("\nExecuting attack...")
    obs, reward, done, truncated, info = env.step(11)  # Attack action
    
    print(f"Action state: {obs['action_state']}")
    print(f"Animation phase: {obs['animation_phase']}")
    print(f"Action progress: {obs['action_progress']:.2f}")
    print(f"Can interrupt: {obs['can_interrupt']}")
    
    env.close()

def demonstrate_stepping_strategies():
    """Demonstrate different stepping strategies"""
    print("\n=== Stepping Strategies ===")
    
    strategies = ['pause', 'continuous', 'adaptive']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        env = EldenGymEnv(
            stepping_strategy=strategy,
            observation_sampling='immediate'
        )
        
        obs, info = env.reset()
        
        # Execute a dodge roll
        start_time = time.time()
        obs, reward, done, truncated, info = env.step(6)  # Dodge forward
        end_time = time.time()
        
        print(f"  Action completed in {end_time - start_time:.2f}s")
        print(f"  Final animation phase: {obs['animation_phase']}")
        
        env.close()

def demonstrate_observation_sampling():
    """Demonstrate different observation sampling strategies"""
    print("\n=== Observation Sampling Strategies ===")
    
    strategies = ['immediate', 'completion', 'smart']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} sampling:")
        env = EldenGymEnv(
            observation_sampling=strategy,
            agent_frequency=2.0  # 2 Hz for testing
        )
        
        obs, info = env.reset()
        
        # Execute multiple actions
        for i in range(3):
            action = 11 if i % 2 == 0 else 6  # Alternate attack and dodge
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"  Step {i+1}: Action={action}, State={obs['action_state']}, "
                  f"Phase={obs['animation_phase']}, Reward={reward}")
            
            if done:
                break
        
        env.close()

def demonstrate_interruption_logic():
    """Demonstrate action interruption logic"""
    print("\n=== Action Interruption Logic ===")
    
    env = EldenGymEnv(
        action_interruption_enabled=True,
        observation_sampling='immediate'
    )
    
    obs, info = env.reset()
    
    # Start a long action (weapon art)
    print("Starting weapon art (2.5s duration)...")
    obs, reward, done, truncated, info = env.step(13)  # Weapon art
    
    print(f"Action state: {obs['action_state']}")
    print(f"Animation phase: {obs['animation_phase']}")
    print(f"Can interrupt: {obs['can_interrupt']}")
    
    # Try to interrupt during startup (should fail)
    print("\nTrying to interrupt during startup...")
    obs, reward, done, truncated, info = env.step(6)  # Dodge
    print(f"Action state after interruption attempt: {obs['action_state']}")
    print(f"Animation phase: {obs['animation_phase']}")
    
    # Wait a bit and try during recovery (should succeed)
    time.sleep(1.0)
    print("\nTrying to interrupt during recovery...")
    obs, reward, done, truncated, info = env.step(6)  # Dodge
    print(f"Action state after interruption attempt: {obs['action_state']}")
    print(f"Animation phase: {obs['animation_phase']}")
    
    env.close()

def demonstrate_frequency_mismatch():
    """Demonstrate handling of agent frequency mismatch"""
    print("\n=== Frequency Mismatch Handling ===")
    
    # Agent slower than game
    print("Testing slow agent (1 Hz) with fast game...")
    env = EldenGymEnv(
        agent_frequency=1.0,  # 1 Hz agent
        observation_sampling='smart',
        stepping_strategy='continuous'
    )
    
    obs, info = env.reset()
    
    for i in range(5):
        action = 11  # Attack
        start_time = time.time()
        obs, reward, done, truncated, info = env.step(action)
        end_time = time.time()
        
        print(f"Step {i+1}: {end_time - start_time:.2f}s, "
              f"State={obs['action_state']}, Reward={reward}")
        
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    print("Elden Ring Advanced Timing Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_animation_phases()
        demonstrate_stepping_strategies()
        demonstrate_observation_sampling()
        demonstrate_interruption_logic()
        demonstrate_frequency_mismatch()
        
        print("\n" + "=" * 50)
        print("Demonstration completed!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure the Elden Ring game is running and the siphon client is connected.")
