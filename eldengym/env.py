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
        action_mode='discrete', # ['discrete', 'multi_binary']
        action_persistence=True,
        action_map=None,
        stepping_frequency=5, # in Hz
        reward_function=None,
        freeze_min_game_speed=1e-5 # Zero breaks game
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.client = EldenClient(host)
        self.action_mode = action_mode
        self.action_persistence = action_persistence
        self.stepping_frequency = stepping_frequency
        self.time_step = 1 / stepping_frequency
        self.freeze_min_game_speed = freeze_min_game_speed

        # Reward function (user-provided or default)
        self.reward_function = reward_function or ScoreDeltaReward(score_key='player_hp')
        if not isinstance(self.reward_function, RewardFunction):
            raise TypeError("reward_fn must inherit from RewardFunction")
        
        # Actions mapping and space
        if action_mode == 'discrete':
            self.action_map = self._discrete_action_map()
            self.action_space = gym.spaces.Discrete(len(self.action_map))
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

    def discrete_action_label(self, action):
        return {
            0: 'no-op',
            1: 'forward', 
            2: 'backward',
            3: 'left',
            4: 'right',
            5: 'jump', 
            6: 'dodge_forward', 
            7: 'dodge_backward', 
            8: 'dodge_left', 
            9: 'dodge_right', 
            10: 'interact', 
            11: 'attack', 
            12: 'use_item'}
    
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
            10: 'LEFT'
            11: 'F',
        }

    def _discrete_action_map(self):
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
        self.client.set_game_speed(1.0)
        self.client.reset_game()
        self.client.start_scenario(self.scenario_name)
        sleep(1) # FIXME: This is a hack to wait for the fight to start.
        self._prev_info = None
        
        # Get initial observation
        obs = self._get_observation()
        
        info = self._get_info()
        self._prev_info = info.copy()
        self.client.set_game_speed(self.freeze_min_game_speed)

        
        return obs, info
    
    def step(self, action):
        # Send actions to server
        # TODO: Implement action persistence
        if self.action_mode == 'discrete':
            keys = self.action_map[action]
            self.client.set_game_speed(1.0)
            sleep(0.001)
            if keys != []:
                self.client.send_key(*keys)
            sleep(self.time_step)
            self.client.set_game_speed(self.freeze_min_game_speed)
        elif self.action_mode == 'multi_binary':
            # TODO: Implement multi-binary action
            pass
        
        # new state
        obs = self._get_observation()
        info = self._get_info()

        # Calculate reward 
        reward = self.reward_function.calculate(obs, info, self._prev_info)
        
        # Check termination 
        terminated = self.reward_function.is_done(obs, info)
        truncated = False

        # Update previous info
        self._prev_info = info.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get observation from server"""
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