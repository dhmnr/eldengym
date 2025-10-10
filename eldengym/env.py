import gymnasium as gym
import numpy as np
from .client.elden_client import EldenClient
from .rewards import RewardFunction, ScoreDeltaReward
from time import sleep

class EldenGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        scenario_name='Margit',
        host='localhost:50051',
        reward_fn=None,
        action_map=None,
        render_mode=None
    ):
        super().__init__()
        
        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.render_mode = render_mode
        
        # Reward function (user-provided or default)
        self.reward_fn = reward_fn or ScoreDeltaReward(score_key='score')
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
    
    def _default_action_map(self):
        """Default keyboard action mapping"""
        return {
            'no-op': [],                    
            'forward': ['w'],                 
            'backward': ['s'],                
            'left': ['a'],                 
            'right': ['d'],                 
            'jump': ['space'],            
            'dodge/sprint': ['shift'],             
            'dodge/sprint_forward': ['w', 'shift'],       
            'dodge/sprint_backward': ['s', 'shift'],        
            'dodge/sprint_left': ['a', 'shift'],       
            'dodge/sprint_right': ['d', 'shift'],       
            'interact': ['e'],                
            'left_click': ['left'],              
            'right_click': ['right'],             
            'middle_click': ['middle'],            
            'use_item': ['r'],            
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.client.reset_game()
        self.client.start_scenario(self.scenario_name)
        sleep(1) # FIXME: This is a hack to wait for the fight to start.
        self._prev_info = None
        
        # Get initial observation
        obs = self._get_observation()
        
        info = self._get_info()
        self._prev_info = info.copy()
        
        return obs, info
    
    def step(self, action):
        # Send keys to server
        keys = self.action_map[action]
        self.client.send_keys(keys)
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward using user's function
        reward = self.reward_fn.calculate(obs, info, self._prev_info)
        
        # Check termination using user's function
        terminated = self.reward_fn.is_done(obs, info)
        truncated = False
        
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
    
    
    def close(self):
        self.client.close()