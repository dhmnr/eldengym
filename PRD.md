# PRD
## Env action Interface
Discrete actions with persistence (persistence = Do not timebox keyboard inputs to env steps, change when actions change)
```
env = EldenRingEnv(
    game_interface=game,
    boss_name='margit',
    action_mode='discrete',
    action_persistence=True,
    timestep_ms=200
)
```
Multi-binary 
```
env = EldenRingEnv(
    game_interface=game,
    boss_name='margit',
    action_mode='multi_binary',
    action_persistence=True,
    timestep_ms=200
)
```
No persistence (jittery movement)
```
env = EldenRingEnv(
    game_interface=game,
    boss_name='margit',
    action_mode='discrete',
    action_persistence=False,  # Compare with/without
    timestep_ms=200
)
```
## Env observations
```
observation_space = gym.spaces.Dict({
    # Visual
    'frame': Box(low=0, high=255, shape=(84, 84, 3), dtype=uint8),
    
    # Game State
    'player_hp': Box(low=0, high=1, shape=(1,), dtype=float32),
    'boss_hp': Box(low=0, high=1, shape=(1,), dtype=float32),
    'distance': Box(low=0, high=20, shape=(1,), dtype=float32),
    
    # Animation States
    'player_animation': Discrete(100),  # Animation ID
    'boss_animation': Discrete(200),
})
```
Optional:
```
  'frame_stack': Box(low=0, high=255, shape=(4, 84, 84, 3), dtype=uint8),
    'action_history': Box(low=0, high=12, shape=(4,), dtype=int32),
```
## Frame preprocessing
 - Cropped/scaled/stacked(temporal)
 - Custom resolution

## Reward function 
 - Base reward function
 - Overriddeble by curriculum wrapper/etc

## Episode Termination
```
done = (
    player_hp <= 0 or              # Player died
    boss_hp <= 0 or                # Boss died
    current_step >= max_steps      # Timeout (900 steps = 3 min default)
)
```

## Info Dict
```
info = {
    'episode_length': int,         # Number of steps
    'episode_reward': float,       # Total reward
    'win': bool,                   # Boss defeated
    'boss_damage_dealt': float,    # % boss HP removed
    'player_damage_taken': float,  # % player HP lost
}
```
## Performance Requirements
TBD

## Vectorized Env support 
TBD


