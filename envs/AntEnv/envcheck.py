
import numpy as np
from envs.antenv import EnvWithGoal
from envs.antenv.create_maze_env import create_maze_env
seed = 1

"""
env generates goal at random feasible state
test env generates goal at (0,16)
"""
env = EnvWithGoal(create_maze_env('AntMaze', seed), 'AntMaze')
test_env = EnvWithGoal(create_maze_env('AntMaze', seed), 'AntMaze')
test_env.evaluate = True # set test goal = (0,16)

env.distance_threshold = 0.5 # set success threshold
env.horizon = 500 # set horizon

observation = env.reset()
ob = observation['observation']
ag = observation['achieved_goal']
bg = observation['desired_goal']
env.early_stop = True # terminates if the agent achieves the goal

for i in range(600):
    action = np.random.uniform(low=-1, high=1, size=8)
    observation, reward, done, info = env.step(action)
    print(observation['achieved_goal'], observation['desired_goal'], reward, info['is_success'])
    env.render()
    if done:
        break