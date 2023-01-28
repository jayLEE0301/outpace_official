from turtle import backward, forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os
import random


class ReplayBuffer(object):
    state_idx_dict = {'tabletop_manipulation' : 6,
                      'sawyer_door' : 7,
                      'sawyer_peg' : 7,
                      }

    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, 
                traj_length = None, sample_type=None, env_name = None,                
                
                ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device
        
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.last_save = 0


        self.sample_type = sample_type
        self.env_name = env_name
        self.traj_length = traj_length
        if traj_length is not None:            
            self.trajwise_capacity = int(capacity/traj_length)
            self.observes_traj = np.empty((self.trajwise_capacity, traj_length, *obs_shape), dtype=np.float32)        
            self.observes_traj_idx = 0
            self.trajwise_full = False
            self.episode_observes = []
        
    
    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        

    # should be called outside
    def add_trajectory(self, episode_observes):
        assert type(episode_observes) is list
        if self.traj_length is not None:
            self.episode_observes = episode_observes #.append(obs)            
            np.copyto(self.observes_traj[self.observes_traj_idx], np.stack(self.episode_observes, axis =0)) #[ts, dim]
            self.observes_traj_idx = (self.observes_traj_idx + 1) % self.trajwise_capacity
            self.trajwise_full = self.trajwise_full or self.observes_traj_idx==0
            self.episode_observes = []
        

    def sample_trajwise_observation(self, batch_size, sample_type=None):
        idxs = np.random.randint(0,
                                 self.trajwise_capacity if self.trajwise_full else self.observes_traj_idx,
                                 size=batch_size)
        if sample_type is None :
            sample_type = self.sample_type

        if sample_type=='only_state':
            obses = torch.as_tensor(self.observes_traj[idxs, :, :self.state_idx_dict[self.env_name]], device=self.device).float() #[bs, ts, dim]
        elif sample_type=='with_initial_state':
            pure_obs = self.observes_traj[idxs, :self.state_idx_dict[self.env_name]]
            init_state = self.observes_traj[idxs, -self.state_idx_dict[self.env_name]:]
            obses = torch.as_tensor(np.concatenate([pure_obs, init_state], axis =-1), device=self.device).float() #[bs, ts, dim]            
        else:
            obses = torch.as_tensor(self.observes_traj[idxs], device=self.device).float() #[bs, ts, dim]
        return obses

    def get_random_indices(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        return idxs

    def sample(self, batch_size, discount, idxs = None):
        # idxs = np.random.randint(0,
        #                          self.capacity if self.full else self.idx,
        #                          size=batch_size)
        if idxs is None:
            idxs = self.get_random_indices(batch_size)
        
        
        if self.sample_type=='only_state':
            # assume original state is concatenated one.
            obses = torch.as_tensor(self.obses[idxs, :self.state_idx_dict[self.env_name]], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs, :self.state_idx_dict[self.env_name]],
                                        device=self.device).float()
        elif self.sample_type=='with_initial_state':
            # assume original state is concatenated one.
            pure_obs = self.obses[idxs, :self.state_idx_dict[self.env_name]]
            init_state = self.obses[idxs, -self.state_idx_dict[self.env_name]:]
            pure_next_obs = self.next_obses[idxs, :self.state_idx_dict[self.env_name]]
            init_next_state = self.next_obses[idxs, -self.state_idx_dict[self.env_name]:]
            obses = torch.as_tensor(np.concatenate([pure_obs, init_state], axis =-1), device=self.device).float()
            next_obses = torch.as_tensor(np.concatenate([pure_next_obs, init_next_state], axis =-1), device=self.device).float()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(self.next_obses[idxs],
                                        device=self.device).float()
                
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        discounts = np.ones((idxs.shape[0], 1), dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)
        dones = torch.as_tensor(~self.not_dones[idxs].astype(bool), device=self.device)

        return obses, actions, rewards, next_obses, discounts, dones

    def sample_all_data(self):
        
        return dict(observation=self.obses,
                    action = self.actions,
                    reward = self.rewards,
                    next_observation=self.next_obses,
                    not_done = self.not_dones,
                    idx = self.idx,
                    full = self.full,
                    )

    def sample_without_relabeling(self, batch_size, discount, sample_only_state = True):
        # should be called in forward gcrl buffer ()
        obses, actions, rewards, next_obses, discounts, dones = self.sample(batch_size, discount)
        if sample_only_state:
            obses = obses[:, :self.state_idx_dict[self.env_name]]
            next_obses = next_obses[:, :self.state_idx_dict[self.env_name]]
        
        return obses, actions, rewards, next_obses, discounts, dones
    
import copy
from enum import Enum
import numpy as np


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}




class HindsightExperienceReplayWrapperVer2(object):
    """
    Wrapper around a replay buffer in order to use HER with memory efficiency.
    Sample relabeled batches when sampling method is called.    
    """

    def __init__(self, replay_buffer, n_sampled_goal, wrapped_env, env_name, consider_done_true = False):
        # super(HindsightExperienceReplayWrapperVer2, self).__init__()
        self.n_sampled_goal = np.inf if n_sampled_goal=='inf' else n_sampled_goal
        self.env = wrapped_env
        self.env_name = env_name
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer
        self._idx_to_future_obs_idx = [None] * self.replay_buffer.capacity

        # for done on success
        self.consider_done_true = consider_done_true

    def add(self, obs_t, action, reward, obs_tp1, done, last_timestep=False):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append(copy.deepcopy((obs_t, action, reward, obs_tp1, done)))
        if last_timestep:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []
            
    def store_episode(self):
        if len(self.episode_transitions) > 0:
            self._store_episode()
            self.episode_transitions = []

    def sample(self, batch_size, *args, **kwargs):        
        indices = self.replay_buffer.get_random_indices(batch_size) 
        
        num_rollout_goals = int(batch_size*1/(self.n_sampled_goal+1)) # batch_size*0.2
        num_future_goals = batch_size - num_rollout_goals # batch_size*0.8
        
        #TODO:should know indices used for sampling
        obses, actions, rewards, next_obses, discounts, dones = self.replay_buffer.sample(None, idxs = indices, *args, **kwargs) # batch_size*0.2
        if self.replay_buffer.sample_type in ['only_state', 'with_initial_state'] : #.sample_only_state:
            # do not relabel
            return obses, actions, rewards, next_obses, discounts

        sample_torch_data = False
        if torch.is_tensor(obses):
            # convert from torch to numpy
            sample_torch_data = True            
            obses = obses.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            rewards = rewards.detach().cpu().numpy()
            next_obses = next_obses.detach().cpu().numpy()
            discounts = discounts.detach().cpu().numpy()
            dones = dones.detach().cpu().numpy()


        obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obses, next_obses)) #TODO: convert should address the batch inputs
                
        if num_future_goals > 0:
            future_indices = indices[-num_future_goals:]
            possible_future_obs_lens = np.array([
                len(self._idx_to_future_obs_idx[i]) for i in future_indices
            ])
            next_obs_idxs = (
                np.random.random(num_future_goals) * possible_future_obs_lens
            ).astype(np.int)
            future_obs_idxs = np.array([
                self._idx_to_future_obs_idx[ids][next_obs_idxs[i]] if self._idx_to_future_obs_idx[ids].shape[0]!=0 else ids # original next_obs idx
                for i, ids in enumerate(future_indices) 
            ]) # idx is global idx in buffer
            assert future_obs_idxs.shape[0]==future_indices.shape[0]
            future_next_obses = self.replay_buffer.next_obses[future_obs_idxs].copy() #[num_future_goals, dim]
            future_next_obses_dict = self.env.convert_obs_to_dict(future_next_obses) #TODO: convert should address the batch inputs
            goal = future_next_obses_dict['achieved_goal'] #[num_future_goals, dim]
            
            obs_dict['desired_goal'][-num_future_goals:] = goal
            next_obs_dict['desired_goal'][-num_future_goals:] = goal
            
            if self.env_name in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
                relabeled_reward = self.env.compute_reward(next_obs_dict['achieved_goal'][-num_future_goals:], goal, sparse = True)             
            elif self.env_name in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
                relabeled_reward = self.env.compute_reward(np.concatenate([next_obs_dict['observation'][-num_future_goals:], goal], axis =-1))
            else:
                relabeled_reward = self.env.compute_reward(np.concatenate([next_obs_dict['achieved_goal'][-num_future_goals:], goal], axis =-1))

            
            # Transform back to ndarrays
            relabeled_obs, relabeled_next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict)) #[batch_size]


            obses = relabeled_obs
            next_obses = relabeled_next_obs
            rewards[-num_future_goals:] = relabeled_reward[:, None] #[num_future_goals] -> [num_future_goals,1]
            
            if self.consider_done_true:
                if np.min(rewards)==-1.: # (-1,0) sparse
                    dones = rewards + 1. # done = True at reward 0 (success)
                else: # (0,1) sparse
                    dones = np.copy(rewards) # done = True at reward 1 (success)

        if sample_torch_data:
            # re-convert from numpy to torch
            obses = torch.as_tensor(obses, device=self.replay_buffer.device).float()
            actions = torch.as_tensor(actions, device=self.replay_buffer.device).float()
            rewards = torch.as_tensor(rewards, device=self.replay_buffer.device).float()
            discounts = torch.as_tensor(discounts, device=self.replay_buffer.device).float()
            next_obses = torch.as_tensor(next_obses, device=self.replay_buffer.device).float()
            dones = torch.as_tensor(dones, device=self.replay_buffer.device).float()
            
        return obses, actions, rewards, next_obses, discounts, dones
        


    
    def __getattr__(self, name):
        return getattr(self.replay_buffer, name)
    

    def __len__(self):
        return len(self.replay_buffer)


    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        episode_length = len(self.episode_transitions)

        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done = transition
            
            # if transition_idx+1 == episode_length -> idx_to_future_obs_idx[current_transition_idx] = np.array([]) (empty)
                
            # TODO : should consider when buffer is full  
            current_transition_idx = copy.deepcopy(self.replay_buffer.idx)
            remained_timesteps_in_current_episode = episode_length - transition_idx -1
            
            if current_transition_idx+1+remained_timesteps_in_current_episode >self.replay_buffer.capacity:
                # should consider when buffer is full  
                # if current_transition_idx ==999999, -> rear : empty , then next current_transition_idx ==0, escape if lines.
                # if current_transition_idx+1+remained_timesteps_in_current_episode ==1000001, (first time if lines is true), 
                future_obs_indices_rear = np.arange(current_transition_idx+1, self.replay_buffer.capacity) #e.g. [10]
                future_obs_indices_front = np.arange(0, remained_timesteps_in_current_episode - future_obs_indices_rear.shape[0]) # [39]

                self._idx_to_future_obs_idx[current_transition_idx] = np.concatenate([future_obs_indices_rear, future_obs_indices_front], axis=0)
            else:
                self._idx_to_future_obs_idx[current_transition_idx] = np.arange(current_transition_idx+1, current_transition_idx+1+remained_timesteps_in_current_episode)
            
            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)
            
            
            

    def sample_all_data(self):
        return self.replay_buffer.sample_all_data()


