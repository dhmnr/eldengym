import gymnasium as gym
import numpy as np
from .client import SiphonClient
from .rewards import RewardFunction, ScoreDeltaReward

class EldenGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        host='localhost:50051',
        memory_addresses=None,
        reward_fn=None,
        action_map=None,
        render_mode=None
    ):
        super().__init__()
        
        self.client = SiphonClient(host)
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
            0: [],                    # no-op
            1: ['w'],                 # up
            2: ['s'],                 # down
            3: ['a'],                 # left
            4: ['d'],                 # right
            5: ['space'],             # jump/action
            6: ['shift'],             # sprint/dodge
            7: ['shift', 'w'],        # sprint forward/dodge forward
            8: ['shift', 's'],        # sprint backward/dodge backward
            9: ['shift', 'a'],        # sprint left/dodge left
            10: ['shift', 'd'],       # sprint right/dodge right
            11: ['e'],                # interact
            12: ['lmb'],              # left click
            13: ['rmb'],              # right click
            14: ['mmb'],              # middle click
            15: ['r'],                # use item
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.client.reset()
        self._prev_info = None
        
        # Get initial observation
        obs = self._get_observation()
        
        # Set observation space if not yet defined
        if self.observation_space is None:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=obs.shape,
                dtype=np.uint8
            )
        
        info = self._get_info()
        self._prev_info = info.copy()
        
        return obs, info
    
    def step(self, action):
        # Send keys to server
        keys = self.action_map[action]
        self.client.send_keys(keys)
        
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
        self._current_frame = frame
        return frame
    
    def _get_info(self):
        """Read memory values"""
        if not hasattr(self, 'memory_addresses') or self.memory_addresses is None:
            return {}
        
        attributeNames = list(self.memory_addresses.keys())
        values = self.client.get_attribute(attributeNames)
        
        return values
    
    def render(self):
        """Render the current frame."""
        if self.render_mode == 'rgb_array':
            return self._current_frame
        elif self.render_mode == 'human':
            # Could add cv2.imshow here
            return self._current_frame
    
    def close(self):
        self.client.close()