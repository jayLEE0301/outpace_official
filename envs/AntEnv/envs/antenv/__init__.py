from turtle import distance
from importlib_metadata import metadata
import numpy as np
import argparse
from collections import deque
from gym import spaces

#import create_maze_env

def get_success_fn(env_name): # we don't use this function
    if env_name in ['AntMaze', 'AntMazeSmall-v0', 'AntMazeComplex-v0', 'AntPush', 'AntFall']:
        return lambda reward: reward > -0.5
    else:
        assert False, 'Unknown env'

class EnvWithGoal(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = 0.5
        self.count = 0
        self.early_stop = False
        self.early_stop_flag = False
        self.horizon = 500
        self.spec = None
        self.metadata = None
        
        obs = self.reset()

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

    def set_attribute(self, evaluate=None, distance_threshold=None, horizon=None, early_stop=None):
        if evaluate is not None:
            self.evaluate = evaluate
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        if horizon is not None:
            self.horizon = horizon
        if early_stop is not None:
            self.early_stop = early_stop
        

    def seed(self, seed):
        self.base_env.seed(seed)

    def rand_goal(self):
        if self.env_name == 'AntMaze':
            while True:
                self.goal = np.random.uniform(low=-4., high=20., size=2)
                if not ((self.goal[0] < 12) and (self.goal[1] > 4) and (self.goal[1] < 12)):
                    break
        elif self.env_name == 'AntMazeSmall-v0':
            while True:
                self.goal = np.random.uniform(low=-2., high=10., size=2)
                if not ((self.goal[0] < 6) and (self.goal[1] > 2) and (self.goal[1] < 6)):
                    break
        elif self.env_name == 'AntMazeComplex-v0':
            while True:
                self.goal = np.array([0.,0.])
                self.goal[0] = np.random.uniform(low=-2., high=10.)
                self.goal[1] = np.random.uniform(low=-2., high=18.)
                if (not ((self.goal[0] < 6) and (self.goal[1] > 2) and (self.goal[1] < 6))) and \
                    (not ((self.goal[0] > 2) and (self.goal[1] > 10) and (self.goal[1] < 14))):
                    break
        else:
            raise NameError('rand goal error')

    def reset(self, goal = None):
        self.early_stop_flag = False
        obs = self.base_env.reset()
        self.count = 0
        if self.evaluate:
            if self.env_name == 'AntMaze':
                self.goal = np.array([0., 16.])
            elif self.env_name == 'AntMazeSmall-v0':
                self.goal = np.array([0., 8.])
            elif self.env_name == 'AntMazeComplex-v0':
                self.goal = np.array([8., 16.])
            else:
                raise NameError('rand goal error')
        else:
            self.rand_goal()
        

        if goal is not None:
            self.goal = goal
            

        self.set_goal("goal_point")
        self.desired_goal = self.goal
        #self.set_camera(obs[:2]) #for skewfit
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }
        reward = self.compute_reward(next_obs['achieved_goal'], next_obs['desired_goal'], sparse=True)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        info['is_success'] = self.success_fn(reward)
        done = self.early_stop_flag or self.count >=  self.horizon
        #self.set_camera(next_obs['achieved_goal']) #for skewfit
        return next_obs, reward, done, info


    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)
    
    def set_goal(self, name):
        body_ids = self.base_env.wrapped_env.model.body_name2id(name)
        

        self.base_env.wrapped_env.model.body_pos[body_ids][:2] = self.goal
        self.base_env.wrapped_env.model.body_quat[body_ids] = [1., 0., 0., 0.]
        

    def compute_reward(self, achieved_goal, goal, sparse=True, threshold = None):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            if threshold is None:
                rs = (np.array(dist) > self.distance_threshold)
            else:
                rs = (np.array(dist) > threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
        return dist


    @property
    def action_space(self):
        return self.base_env.action_space

    def reset_goal(self, goal):
        self.desired_goal = goal.copy()
        self.goal = goal.copy()
        self.set_goal("goal_point")