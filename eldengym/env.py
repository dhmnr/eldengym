import gymnasium as gym
import numpy as np
from .client.elden_client import EldenClient
from .rewards import RewardFunction, ScoreDeltaReward
from time import sleep, time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

class ActionState(Enum):
    """Track the state of actions"""
    IDLE = "idle"
    EXECUTING = "executing" 
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"

class ActionType(Enum):
    """Categorize actions by their properties"""
    INSTANT = "instant"  # No-op, camera movement
    MOVEMENT = "movement"  # Walking, running (interruptible)
    COMBAT = "combat"  # Attacks, spells (non-interruptible)
    DODGE = "dodge"  # Rolling, jumping (semi-interruptible)

class AnimationPhase(Enum):
    """Track animation phases for precise interruption"""
    STARTUP = "startup"  # Wind-up phase (non-interruptible)
    ACTIVE = "active"    # Hit/effect phase (non-interruptible) 
    RECOVERY = "recovery" # Backswing phase (interruptible)
    IDLE = "idle"        # No animation

class EldenGymEnv(gym.Env):
    
    def __init__(
        self,
        scenario_name='margit',
        host='localhost:50051',
        action_mode='discrete', # ['discrete', 'multi_binary']
        action_map=None,
        observation_mode='frame', # ['frame', 'frame_stack']
        frame_stack_size=4,
        stepping_mode='continuous', # ['continuous', 'discrete']
        stepping_interval=0.02, # in seconds
        reward_function=None,
        max_action_duration=3.0,  # Maximum time to wait for action completion
        action_interruption_enabled=True,  # Allow interrupting certain actions
        stepping_strategy='adaptive',  # ['pause', 'continuous', 'adaptive']
        agent_frequency=10.0,  # Expected agent decision frequency (Hz)
        observation_sampling='smart',  # ['immediate', 'completion', 'smart']
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.frame_stack_size = frame_stack_size
        self.stepping_mode = stepping_mode
        self.stepping_interval = stepping_interval
        self.max_action_duration = max_action_duration
        self.action_interruption_enabled = action_interruption_enabled
        self.stepping_strategy = stepping_strategy
        self.agent_frequency = agent_frequency
        self.observation_sampling = observation_sampling

        # Reward function (user-provided or default)
        self.reward_function = reward_function or ScoreDeltaReward(score_key='player_hp')
        if not isinstance(self.reward_function, RewardFunction):
            raise TypeError("reward_fn must inherit from RewardFunction")
        
        # Actions mapping and space
        if action_mode == 'discrete':
            self.action_map = self._discrete_action_map()
            self.action_space = gym.spaces.Discrete(len(self.action_map))
            self.action_keybindings = self._discrete_action_keybindings()
        elif action_mode == 'multi_binary':
            self.action_map = self._multi_binary_action_map()
            self.action_space = gym.spaces.MultiBinary(len(self.action_map))
        else:
            raise ValueError(f"Invalid action mode: {action_mode}")
        
        
        # Observation space 
        self._obs_shape = None
        self.observation_space = None
        
        # State tracking
        self._prev_info = None
        self._current_frame = None
        
        # Action execution tracking
        self._current_action = None
        self._action_start_time = None
        self._action_state = ActionState.IDLE
        self._action_queue = []
        self._last_animation_id = None
        self._current_animation_phase = AnimationPhase.IDLE
        self._last_step_time = None
        self._agent_step_interval = 1.0 / self.agent_frequency
    
    def _multi_binary_action_map(self):
        return {
            0: '',
            1: 'W',
            2: 'A',
            3: 'S',
            4: 'D',
            5: 'SPACE',
            6: 'LEFT_SHIFT',
            7: 'E',
            8: 'LEFT_ALT',
            9: 'R',
            10: 'LEFT',
            11: 'F',
        }

    def _discrete_action_map(self):
        """Default discrete action mapping with action metadata and animation phases"""
        return {
            0: {'name': 'no-op', 'type': ActionType.INSTANT, 'duration': 0.0, 'interruptible': True, 'phases': {}},
            1: {'name': 'forward', 'type': ActionType.MOVEMENT, 'duration': 0.1, 'interruptible': True, 'phases': {}}, 
            2: {'name': 'backward', 'type': ActionType.MOVEMENT, 'duration': 0.1, 'interruptible': True, 'phases': {}},
            3: {'name': 'left', 'type': ActionType.MOVEMENT, 'duration': 0.1, 'interruptible': True, 'phases': {}},
            4: {'name': 'right', 'type': ActionType.MOVEMENT, 'duration': 0.1, 'interruptible': True, 'phases': {}},
            5: {'name': 'jump', 'type': ActionType.DODGE, 'duration': 0.8, 'interruptible': False, 
                'phases': {'startup': 0.2, 'active': 0.3, 'recovery': 0.3}}, 
            6: {'name': 'dodge_forward', 'type': ActionType.DODGE, 'duration': 1.2, 'interruptible': False,
                'phases': {'startup': 0.1, 'active': 0.4, 'recovery': 0.7}}, 
            7: {'name': 'dodge_backward', 'type': ActionType.DODGE, 'duration': 1.2, 'interruptible': False,
                'phases': {'startup': 0.1, 'active': 0.4, 'recovery': 0.7}}, 
            8: {'name': 'dodge_left', 'type': ActionType.DODGE, 'duration': 1.2, 'interruptible': False,
                'phases': {'startup': 0.1, 'active': 0.4, 'recovery': 0.7}}, 
            9: {'name': 'dodge_right', 'type': ActionType.DODGE, 'duration': 1.2, 'interruptible': False,
                'phases': {'startup': 0.1, 'active': 0.4, 'recovery': 0.7}}, 
            10: {'name': 'interact', 'type': ActionType.INSTANT, 'duration': 0.5, 'interruptible': True, 'phases': {}}, 
            11: {'name': 'attack', 'type': ActionType.COMBAT, 'duration': 1.5, 'interruptible': False,
                'phases': {'startup': 0.4, 'active': 0.3, 'recovery': 0.8}}, 
            12: {'name': 'use_item', 'type': ActionType.COMBAT, 'duration': 2.0, 'interruptible': False,
                'phases': {'startup': 0.5, 'active': 0.5, 'recovery': 1.0}},
            13: {'name': 'weapon_art', 'type': ActionType.COMBAT, 'duration': 2.5, 'interruptible': False,
                'phases': {'startup': 0.8, 'active': 0.7, 'recovery': 1.0}}
        }

    
    def _discrete_action_keybindings(self):
        """Default discrete action keybindings"""
        return {
            'no-op': [],                    
            'forward': [['W'], 500, 0],                 
            'backward': [['S'], 500, 0],                
            'left': [['A'], 500, 0],                 
            'right': [['D'], 500, 0],                 
            'jump': [['SPACE'], 500, 0],            
            'dodge_forward': [['W', 'LEFT_SHIFT'], 100, 200],       
            'dodge_backward': [['S', 'LEFT_SHIFT'], 100, 200],        
            'dodge_left': [['A', 'LEFT_SHIFT'], 100, 200],       
            'dodge_right': [['D', 'LEFT_SHIFT'], 100, 200],       
            'interact': [['E'], 500, 0],                
            'attack': [['LEFT_ALT', 'LEFT'], 400, 0],        
            'use_item': [['R'], 500, 0],            
            'weapon_art': [['F'], 500, 0],        
        }
    
    def _get_current_animation_phase(self) -> AnimationPhase:
        """Determine current animation phase based on elapsed time"""
        if self._action_state != ActionState.EXECUTING or self._current_action is None:
            return AnimationPhase.IDLE
            
        action_data = self.action_map[self._current_action]
        elapsed = time() - self._action_start_time
        
        if not action_data['phases']:
            return AnimationPhase.IDLE
            
        # Determine phase based on elapsed time
        if elapsed < action_data['phases'].get('startup', 0):
            return AnimationPhase.STARTUP
        elif elapsed < action_data['phases'].get('startup', 0) + action_data['phases'].get('active', 0):
            return AnimationPhase.ACTIVE
        elif elapsed < action_data['duration']:
            return AnimationPhase.RECOVERY
        else:
            return AnimationPhase.IDLE
    
    def _can_interrupt_action(self, new_action_id: int) -> bool:
        """Check if current action can be interrupted by new action based on animation phase"""
        if self._action_state != ActionState.EXECUTING:
            return True
            
        current_action = self.action_map[self._current_action]
        new_action = self.action_map[new_action_id]
        current_phase = self._get_current_animation_phase()
        
        # Can't interrupt during startup or active phases
        if current_phase in [AnimationPhase.STARTUP, AnimationPhase.ACTIVE]:
            return False
            
        # Can interrupt during recovery phase for certain action types
        if current_phase == AnimationPhase.RECOVERY:
            if current_action['type'] == ActionType.MOVEMENT:
                return True
            elif current_action['type'] in [ActionType.DODGE, ActionType.COMBAT]:
                # Only allow interruption by higher priority actions
                return new_action['type'] in [ActionType.DODGE, ActionType.COMBAT]
                
        return True
    
    def _execute_action(self, action_id: int) -> bool:
        """Execute an action and return whether it was successful"""
        action_data = self.action_map[action_id]
        action_name = action_data['name']
        
        if action_name == 'no-op':
            return True
            
        # Get keybindings for this action
        keybinding = self.action_keybindings.get(action_name, [])
        if not keybinding:
            return False
            
        # Send the key combination
        self.client.send_key(*keybinding)
        return True
    
    def _is_action_complete(self) -> bool:
        """Check if current action is complete based on animation or time"""
        if self._action_state != ActionState.EXECUTING:
            return True
            
        current_time = time()
        action_data = self.action_map[self._current_action]
        
        # Check if action duration has passed
        if current_time - self._action_start_time >= action_data['duration']:
            return True
            
        # Check animation change for combat actions
        if action_data['type'] == ActionType.COMBAT:
            current_animation = self.client.player_animation_id
            if current_animation != self._last_animation_id:
                # Animation changed, action might be complete
                return True
                
        return False
    
    def _update_action_state(self):
        """Update the current action state"""
        if self._action_state == ActionState.EXECUTING:
            if self._is_action_complete():
                self._action_state = ActionState.COMPLETED
                self._current_action = None
                self._action_start_time = None
                self._current_animation_phase = AnimationPhase.IDLE
    
    def _should_sample_observation(self) -> bool:
        """Determine if we should sample an observation based on strategy"""
        current_time = time()
        
        if self.observation_sampling == 'immediate':
            return True
        elif self.observation_sampling == 'completion':
            return self._action_state in [ActionState.COMPLETED, ActionState.IDLE]
        elif self.observation_sampling == 'smart':
            # Sample at action completion or agent frequency intervals
            if self._action_state in [ActionState.COMPLETED, ActionState.IDLE]:
                return True
            if (self._last_step_time is None or 
                current_time - self._last_step_time >= self._agent_step_interval):
                return True
        return False
    
    def _handle_stepping_strategy(self):
        """Handle different stepping strategies"""
        if self.stepping_strategy == 'pause':
            # Pause game during action execution
            if self._action_state == ActionState.EXECUTING:
                self.client.set_game_speed(0.1)  # Slow motion
            else:
                self.client.set_game_speed(1.0)
        elif self.stepping_strategy == 'continuous':
            # Let game run at normal speed
            self.client.set_game_speed(1.0)
        elif self.stepping_strategy == 'adaptive':
            # Adaptive speed based on action type and phase
            if self._action_state == ActionState.EXECUTING:
                action_data = self.action_map[self._current_action]
                current_phase = self._get_current_animation_phase()
                
                if current_phase == AnimationPhase.STARTUP:
                    self.client.set_game_speed(0.5)  # Slow during windup
                elif current_phase == AnimationPhase.ACTIVE:
                    self.client.set_game_speed(1.0)  # Normal during active
                elif current_phase == AnimationPhase.RECOVERY:
                    self.client.set_game_speed(0.7)  # Slightly slow during recovery
            else:
                self.client.set_game_speed(1.0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.set_game_speed(1.0)
        self.client.reset_game()
        self.client.start_scenario(self.scenario_name)
        sleep(1) # FIXME: This is a hack to wait for the fight to start.
        
        # Reset action tracking
        self._current_action = None
        self._action_start_time = None
        self._action_state = ActionState.IDLE
        self._action_queue = []
        self._last_animation_id = self.client.player_animation_id
        self._current_animation_phase = AnimationPhase.IDLE
        self._last_step_time = None
        
        self._prev_info = None
        
        # Get initial observation
        obs = self._get_observation()
        
        info = self._get_info()
        self._prev_info = info.copy()
        self.client.set_game_speed(self.freeze_min_game_speed)

        
        return obs, info
    
    def step(self, action):
        """Execute a step with advanced action timing and stepping strategies"""
        current_time = time()
        
        # Update current action state and animation phase
        self._update_action_state()
        self._current_animation_phase = self._get_current_animation_phase()
        
        # Handle new action
        if self.action_mode == 'discrete':
            action_data = self.action_map[action]
            
            # Check if we can interrupt current action
            if self._action_state == ActionState.EXECUTING:
                if not self._can_interrupt_action(action):
                    # Can't interrupt, queue the action or ignore
                    if self.action_interruption_enabled:
                        self._action_queue.append(action)
                    # Continue with current action
                else:
                    # Interrupt current action
                    self._action_state = ActionState.INTERRUPTED
                    self._current_action = None
                    self._action_start_time = None
            
            # Execute new action if we're not busy or can interrupt
            if self._action_state in [ActionState.IDLE, ActionState.COMPLETED, ActionState.INTERRUPTED]:
                if self._execute_action(action):
                    self._current_action = action
                    self._action_start_time = current_time
                    self._action_state = ActionState.EXECUTING
                    self._last_animation_id = self.client.player_animation_id
                    self._current_animation_phase = AnimationPhase.STARTUP
            
            # Handle stepping strategy
            self._handle_stepping_strategy()
            
            # Wait for action to complete or timeout
            if self._action_state == ActionState.EXECUTING:
                action_data = self.action_map[self._current_action]
                max_wait = min(action_data['duration'], self.max_action_duration)
                
                # Wait for action completion with periodic checks
                start_wait = time()
                while (time() - start_wait < max_wait and 
                       self._action_state == ActionState.EXECUTING):
                    self._update_action_state()
                    self._current_animation_phase = self._get_current_animation_phase()
                    self._handle_stepping_strategy()
                    sleep(0.01)  # Small sleep to prevent busy waiting
                
                # If still executing after max wait, force completion
                if self._action_state == ActionState.EXECUTING:
                    self._action_state = ActionState.COMPLETED
                    self._current_action = None
                    self._action_start_time = None
                    self._current_animation_phase = AnimationPhase.IDLE
                
        elif self.action_mode == 'multi_binary':
            # TODO: Implement multi-binary action handling
            pass
        
        # Determine if we should sample observation
        should_sample = self._should_sample_observation()
        
        if should_sample:
            # Get new state
            obs = self._get_observation()
            info = self._get_info()

            # Calculate reward 
            reward = self.reward_function.calculate(obs, info, self._prev_info)
            
            # Check termination 
            terminated = self.reward_function.is_done(obs, info)
            truncated = False

            # Update previous info
            self._prev_info = info.copy()
            self._last_step_time = current_time
            
            return obs, reward, terminated, truncated, info
        else:
            # Return previous observation if not sampling
            obs = self._get_observation()
            info = self._get_info()
            reward = 0.0  # No reward change if not sampling
            terminated = self.reward_function.is_done(obs, info)
            truncated = False
            
            return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get observation from server with action state information"""
        frame = self.client.get_frame()
        
        # Calculate action progress
        action_progress = 0.0
        if self._action_state == ActionState.EXECUTING and self._current_action is not None:
            action_data = self.action_map[self._current_action]
            elapsed = time() - self._action_start_time
            action_progress = min(elapsed / action_data['duration'], 1.0)
        
        return {
            'frame': frame,
            'boss_hp': self.client.target_hp / self.client.target_max_hp,
            'player_hp': self.client.player_hp / self.client.player_max_hp,
            'distance': self.client.target_player_distance,
            'boss_animation': self.client.target_animation_id,
            'player_animation': self.client.player_animation_id,
            'action_state': self._action_state.value,
            'current_action': self._current_action,
            'action_progress': action_progress,
            'action_queue_length': len(self._action_queue),
            'animation_phase': self._current_animation_phase.value,
            'can_interrupt': self._can_interrupt_action(0) if self._current_action is not None else True,
        }
    
    def _get_info(self):
        return {
            'player_hp': self.client.player_hp,
            'action_state': self._action_state.value,
            'current_action': self._current_action,
            'action_queue_length': len(self._action_queue),
            'animation_phase': self._current_animation_phase.value,
            'can_interrupt': self._can_interrupt_action(0) if self._current_action is not None else True,
        }
    
    def close(self):
        self.client.close()