from lib2to3.pytree import convert
from turtle import backward, forward
from xml.etree.ElementInclude import include
import numpy as np
import gym            
from gym.spaces import Box
from scipy.misc import derivative



class StateWrapper(object):
    def __init__(self, env) -> None:
        self.env = env
        
    def action_spec(self):
        return self.env.action_space
    
    def observation_spec(self, option=None):
        if option is None:
            return self.env.observation_space
        elif option=='forward':
            return self.env.forward_observation_space
        elif option=='backward':
            return self.env.backward_observation_space
    
    def __getattr__(self, attrname):
        return getattr(self.env, attrname)




# already unwrapped env
class WraptoGoalEnv(object): 
    '''
    NOTE : Make the env as a goal env
    '''
    
    def __init__(self, env, env_name = None, convert_goal_to_reach_object=False):
        
        self.env = env        
        self.env_name = env_name
        
        self.reduced_key_order = ['observation', 'desired_goal'] # assume observation==achieved_goal
        
        obs = self.env.reset()
        obs_dict = self.convert_obs_to_dict(obs)
        
        self.obs_dim = obs_dict['observation'].shape[0]
        self.goal_dim = obs_dict['desired_goal'].shape[0]
        
        self.convert_goal_to_reach_object = convert_goal_to_reach_object
        
    def convert_dict_to_obs(self, obs_dict, batch_ver=False):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        
        return np.concatenate([obs_dict[key] for key in self.reduced_key_order], axis = -1)
            

    def convert_obs_to_dict(self, obs, batch_ver=False):
        
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        # Currently restricted to FetchEnv
        if 'tabletop' in self.env_name:            
            assert obs.shape[-1]==12, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :6] ,
                "achieved_goal": obs[..., :6] ,
                "desired_goal": obs[..., 6:] ,
            }
            
        elif self.env_name in ['sawyer_peg', 'sawyer_door']:            
            assert obs.shape[-1]==14, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :7] ,
                "achieved_goal": obs[..., :7] ,
                "desired_goal": obs[..., 7:] ,
            }
        elif self.env_name in [    'sawyer_peg_push','sawyer_peg_pick_and_place']:            
            assert obs.shape[-1]==10, 'obs shape is {}'.format(obs.shape)
            return {
                "observation": obs[..., :7] ,
                "achieved_goal": obs[..., 4:7] ,
                "desired_goal": obs[..., -3:] ,
            }
        else:
            raise NotImplementedError

    def is_successful_deviating_initial_state(self, obs):
        if self.env_name=='sawyer_door':
            return np.linalg.norm(obs[..., :7] - self.env.init_state[..., :7], axis =-1) > 0.02
            
        elif self.env_name in ['sawyer_peg']:
            return np.linalg.norm(obs[..., :7] - self.env.init_state[..., :7], axis =-1) >  self.env.TARGET_RADIUS
            
        elif self.env_name=='tabletop':
            return np.linalg.norm(obs[..., :4] - self.env.init_state[..., :4], axis =-1) > 0.2
            
        else:
            raise NotImplementedError
    
    def is_different_init_state_and_goal(self, obs):
        if self.env_name=='sawyer_door':
            return np.linalg.norm(obs[..., 7:14] - self.env.init_state[..., :7], axis =-1) > 0.02
            
        elif self.env_name in ['sawyer_peg']:
            return np.linalg.norm(obs[..., 7:14] - self.env.init_state[..., :7], axis =-1) >  self.env.TARGET_RADIUS
            
        elif self.env_name=='tabletop':
            return np.linalg.norm(obs[..., 6:10] - self.env.init_state[..., :4], axis =-1) > 0.2
            
        else:
            raise NotImplementedError

    
    def compute_reward(self, obs):        
        # Assume sparse reward!
        return (self.is_successful(obs=obs)).astype(np.float)
        

    def is_successful(self, obs):
        if self.convert_goal_to_reach_object:                        
            raise NotImplementedError
        else:
            if self.env_name=='sawyer_door':
                return np.linalg.norm(obs[..., 4:7] - obs[..., 11:14], axis =-1) <= 0.02
            elif self.env_name in ['sawyer_peg']:
                return np.linalg.norm(obs[..., 4:7] - obs[..., 11:14], axis =-1) <= self.env.TARGET_RADIUS
            elif self.env_name in [   'sawyer_peg_push','sawyer_peg_pick_and_place']:                
                return np.linalg.norm(obs[..., 4:7] - obs[..., -3:], axis =-1) <= self.env.TARGET_RADIUS
            elif self.env_name=='tabletop_manipulation':
                return np.linalg.norm(obs[..., :4] - obs[..., 6:-2], axis =-1) <= 0.2
            else:
                raise NotImplementedError
    

    def get_hand_pos(self, obs):
        if self.env_name=='sawyer_door':
            return obs[..., :3]
        elif self.env_name in ['sawyer_peg',    'sawyer_peg_push','sawyer_peg_pick_and_place']:
            return obs[..., :3]
        elif self.env_name=='tabletop_manipulation':
            return obs[..., :2]
        elif 'Fetch' in self.env_name:
            return obs[..., :3]
        elif 'Ant' in self.env_name:
            raise NotImplementedError

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)





from collections import OrderedDict
import numpy as np
from gym import spaces
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.

    :param env: (gym.GoalEnv)
    """

    def __init__(self, env, env_name = None):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.env_name = env_name
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER], axis =-1)

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[..., :self.obs_dim]),
            ('achieved_goal', observations[..., self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[..., self.obs_dim + self.goal_dim:]),
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.convert_dict_to_obs(obs), reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self, *args, **kwargs):
        return self.convert_dict_to_obs(self.env.reset(*args, **kwargs))

    def compute_reward(self, achieved_goal, desired_goal, *args, **kwargs): # info=None,
        return self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()
    
    

    def is_successful(self, obs):
        # for treating the batch inputs
        if self.env_name=='sawyer_door':
            return np.linalg.norm(obs[..., 4:7] - obs[..., 11:14], axis =-1) <= 0.02
        elif self.env_name=='sawyer_peg':
            return np.linalg.norm(obs[..., 4:7] - obs[..., 11:14], axis =-1) <= self.env.TARGET_RADIUS
        elif self.env_name=='tabletop_manipulation':
            return np.linalg.norm(obs[..., :4] - obs[..., 6:-2], axis =-1) <= 0.2
        elif 'Fetch' in self.env_name:
            return np.linalg.norm(obs[..., -6:-3] - obs[..., -3:], axis =-1) <= 0.05
        elif 'Ant' in self.env_name:
            return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= self.env.distance_threshold
        elif 'Maze' in self.env_name:
            return np.linalg.norm(obs[..., -4:-2] - obs[..., -2:], axis =-1) <= self.env.distance_threshold
        else:
            raise NotImplementedError
    

    def get_hand_pos(self, obs):
        if self.env_name=='sawyer_door':
            return obs[..., :3]
        elif self.env_name=='sawyer_peg':
            return obs[..., :3]
        elif self.env_name=='tabletop_manipulation':
            return obs[..., :2]
        elif 'Fetch' in self.env_name:
            return obs[..., :3]
        elif 'Ant' in self.env_name:
            raise NotImplementedError

   

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)

import copy
class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0, earl_env = False, relative_goal_env = False):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        self.earl_env = earl_env
        # self.antmaze_env = antmaze_env
        self.relative_goal_env = relative_goal_env
        if earl_env:
            assert reward_offset==0.0, 'assume earl outputs 0,1 sparse reward'

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.earl_env:
            info.update({'earl_done' : copy.deepcopy(done)})
        done = done or info.get('is_success', False) # True when Timelimit or success or other reasones in original env
        
        if self.relative_goal_env: # want to return done=True only for final goal is achieved, not subgoal            
            info.update({'is_current_goal_success' : info['is_success']}) # for chainging to the next subgoal
            # info.update({'relative_goal_done' : copy.deepcopy(done)})
            if not self.env.is_final_goal: # should be set in reset_goal in RelativeSubGoalWrapper
                done = False
            
        if self.earl_env:
            done  = done or self.env.is_successful(obs)            
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, *args, **kwargs):
        reward = self.env.compute_reward(achieved_goal, desired_goal, *args, **kwargs)
        return reward + self.reward_offset
    

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)




class ResidualGoalWrapper(gym.Wrapper):
    
    def __init__(self, env, env_name):
        super(ResidualGoalWrapper, self).__init__(env)        
        self.env_name = env_name
        
    
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.is_final_goal = False
        self.is_residual_goal = False
        self.original_goal_success = False
        self.residual_goalstep = 0
        return obs
        
    
    def step(self, action):
        # Assume obs_dict is given
        obs, reward, done, info = self.env.step(action)
        if self.is_residual_goal:
            self.residual_goalstep += 1
        return obs, reward, done, info
    


    def reset_goal(self, goal, is_final_goal = False):
        if self.env_name in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0",  'sawyer_peg_push','sawyer_peg_pick_and_place']:
            self.env.reset_goal(goal.copy())
            # self.env.goal = goal.copy()
            # self.env.desired_goal = goal.copy()
        else:
            raise NotImplementedError
        
        self.is_final_goal = is_final_goal
        self.is_residual_goal = True
        self.residual_goalstep = 0
        
    
    def __getattr__(self, attrname):
        return getattr(self.env, attrname)




