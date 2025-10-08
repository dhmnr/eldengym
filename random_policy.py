from gym_grpc_env import GRPCGameEnv
from gym_grpc_env.rewards import RewardFunction
from gym_grpc_env.wrappers import FrameStack, ResizeFrame

# Define custom reward function
class MarioReward(RewardFunction):
    def calculate(self, obs, info, prev_info=None):
        if prev_info is None:
            return 0.0
        
        # Reward for score increase and position change
        score_reward = info['score'] - prev_info['score']
        pos_reward = (info['x_pos'] - prev_info['x_pos']) * 0.1
        
        # Penalty for losing lives
        life_penalty = (prev_info['lives'] - info['lives']) * -100
        
        return score_reward + pos_reward + life_penalty
    
    def is_done(self, obs, info):
        return info['lives'] <= 0 or info['time'] <= 0

# Create environment
env = GRPCGameEnv(
    host='localhost:50051',
    memory_addresses={
        'score': 0x07E0,
        'lives': 0x075A,
        'x_pos': 0x086,
        'time': 0x07F8
    },
    reward_fn=MarioReward(),
    action_map={
        0: [],           # no-op
        1: ['a'],        # left
        2: ['d'],        # right  
        3: ['space'],    # jump
        4: ['d', 'space'], # run jump
    }
)

# Add wrappers
env = ResizeFrame(env, 84, 84)
env = FrameStack(env, num_stack=4)

# Use it
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

env.close()