import gymnasium as gym
import numpy as np
from .client.elden_client import EldenClient
from .rewards import RewardFunction, ScoreDeltaReward
from time import sleep

class EldenGymEnv(gym.Env):
    
    def __init__(
        self,
        scenario_name='margit',
        host='localhost:50051',
        reward_fn=None,
        action_level='raw', # ['raw', 'semantic']
        action_map=None,
        stepping_logic='fixed', # ['fixed', 'dynamic']
        time_step=0.01,
        freeze_game_speed=1e-5
    ):
        super().__init__()
        
        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.action_level = action_level
        self.stepping_logic = stepping_logic
        self.time_step = time_step
        self.freeze_game_speed = freeze_game_speed
        # Reward function (user-provided or default)
        self.reward_fn = reward_fn or ScoreDeltaReward(score_key='player_hp')
        if not isinstance(self.reward_fn, RewardFunction):
            raise TypeError("reward_fn must inherit from RewardFunction")
        
        # Action mapping (keyboard keys per action)
        self.action_map = action_map or self._default_action_map()
        
        # Define spaces
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        
        # Observation space (will be set on first reset)
        self._obs_shape = None
        self.observation_space = None
        
        # State tracking
        self._prev_info = None
        self._current_frame = None

    def action_to_index(self, action):
        return {
            'no-op': 0, 
            'forward': 1, 
            'backward': 2, 
            'left': 3, 
            'right': 4, 
            'jump': 5, 
            'dodge_forward': 6, 
            'dodge_backward': 7, 
            'dodge_left': 8, 
            'dodge_right': 9, 
            'interact': 10, 
            'attack': 11, 
            'use_item': 12}
    
    def _default_action_map(self):
        """Default keyboard action mapping"""
        return {
            0: [],                    
            1: [['W'], 500, 0],                 
            2: [['S'], 500, 0],                
            3: [['A'], 500, 0],                 
            4: [['D'], 500, 0],                 
            5: [['SPACE'], 500, 0],            
            6: [['W', 'LEFT_SHIFT'], 100, 200],       
            7: [['S', 'LEFT_SHIFT'], 100, 200],        
            8: [['A', 'LEFT_SHIFT'], 100, 200],       
            9: [['D', 'LEFT_SHIFT'], 100, 200],       
            10: [['E'], 500, 0],                
            11: [['LEFT_ALT', 'LEFT'], 400, 0],        
            12: [['R'], 500, 0],            
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.set_game_speed(self.time_step)
        self.client.reset_game()
        self.client.start_scenario(self.scenario_name)
        sleep(1) # FIXME: This is a hack to wait for the fight to start.
        self._prev_info = None
        
        # Get initial observation
        obs = self._get_observation()
        
        info = self._get_info()
        self._prev_info = info.copy()
        self.client.set_game_speed(self.freeze_game_speed)

        
        return obs, info
    
    def step(self, action):
        # Send keys to server
        keys = self.action_map[action]
        self.client.set_game_speed(self.time_step)
        sleep(0.01)
        if keys != []:
            self.client.send_key(*keys)
        sleep(self.time_step)
        self.client.set_game_speed(self.freeze_game_speed)
        # Get new state
        obs = self._get_observation()
        info = self._get_info()

        # Calculate reward using user's function
        reward = self.reward_fn.calculate(obs, info, self._prev_info)
        
        # Check termination using user's function
        terminated = self.reward_fn.is_done(obs, info)
        truncated = False

        # Update previous info
        self._prev_info = info.copy()

        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get processed frame from server"""
        frame = self.client.get_frame()
        return {
            'frame': frame,
            'boss_hp': self.client.target_hp / self.client.target_max_hp,
            'player_hp': self.client.player_hp / self.client.player_max_hp,
            'distance': self.client.target_player_distance,
            'boss_animation': self.client.target_animation_id,
            'player_animation': self.client.player_animation_id,    
        }
    
    def _get_info(self):
        return {
            'player_hp': self.client.player_hp,
        }
    
    def close(self):
        self.client.close()