import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import utils
import hydra
import time
from typing import Union
import warnings
import matplotlib.pyplot as plt

from outpace_train import get_object_states_only_from_goal


def wasserstein_reward(d: torch.Tensor) -> torch.Tensor:
    """
    return the wasserstein reward
    """
    return d

reward_mapping = {'aim': wasserstein_reward,
                }



class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=1, activ=F.relu, output_nonlinearity=None, n_units=64, tanh_constant=1.0):
        super(MlpNetwork, self).__init__()
        
        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
        self.out_nl = output_nonlinearity
        self.activ = activ
        self.tanh_constant = tanh_constant
        self.apply(utils.weight_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of network
        :param x:
        :return:
        """
        x = self.activ(self.h1(x))
        x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == F.log_softmax:
                x = F.log_softmax(x, dim=-1)
            else:
                if self.out_nl==torch.tanh:                    
                    x = self.out_nl(self.tanh_constant*x)
                else:
                    x = self.out_nl(x)
        return x

class DiscriminatorEnsemble(nn.Module):
    def __init__(self, n_ensemble, x_dim=1, reward_type='aim', lr = 1e-4, lipschitz_constant=0.1, output_activation= None, device = 'cuda:0',
                env_name=None, tanh_constant = 1.0, lambda_coef = 10.0, adam_eps=1e-8, optim = 'adam'):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.adam_eps = adam_eps
        self.optim = optim
        self.discriminator_ensemble = nn.ModuleList([Discriminator(x_dim, reward_type, lr, lipschitz_constant, output_activation, device,
                env_name, tanh_constant, lambda_coef, adam_eps, optim) for i in range(n_ensemble)])
                            
        self.apply(utils.weight_init)

    def forward(self, inputs):
        h = inputs
        outputs = torch.stack([discriminator(h) for discriminator in self.discriminator_ensemble], dim = 1) #[bs, n_ensemble, dim(1)]
        outputs = torch.mean(outputs, dim = 1)  #[bs, 1]
        return outputs

    def std(self,inputs):
        aim_outputs = torch.stack(self.forward(inputs), dim = 1)  # [bs, n_ensemble, 1]
        return torch.std(aim_outputs, dim = 1, keepdim=False) #[bs, 1]

    def reward(self, x: torch.Tensor) -> np.ndarray:
        return np.stack([discriminator.reward(x) for discriminator in self.discriminator_ensemble], axis = 1).mean(axis=1)

    def optimize_discriminator(self, *args, **kwargs):        
        loss_list = []
        wgan_loss_list = []
        graph_penalty_list = []
        # min_aim_f_loss_list = []

        for discriminator in self.discriminator_ensemble:
            loss, wgan_loss, graph_penalty, min_aim_f_loss = discriminator.optimize_discriminator(*args, **kwargs)
            loss_list.append(loss)
            wgan_loss_list.append(wgan_loss)
            graph_penalty_list.append(graph_penalty)
            # min_aim_f_loss_list.append(min_aim_f_loss)
        return torch.stack(loss_list, dim = 0).mean(0), torch.stack(wgan_loss_list, dim = 0).mean(0), torch.stack(graph_penalty_list, dim = 0).mean(0), None
    
    


class Discriminator(nn.Module):
    def __init__(self, x_dim=1, reward_type='aim', lr = 1e-4, lipschitz_constant=0.1, output_activation= None, device = 'cuda:0',
                env_name=None, tanh_constant = 1.0, lambda_coef = 10.0, adam_eps=1e-8, optim='adam'):
        self.use_cuda = torch.cuda.is_available()
        self.device = device # torch.device("cuda" if self.use_cuda else "cpu")
        
        self.adam_eps = adam_eps
        self.optim = optim        
        super(Discriminator, self).__init__()
        self.input_dim = x_dim
        assert reward_type in ['aim', 'gail', 'airl', 'fairl']
        self.reward_type = reward_mapping[reward_type]
        if self.reward_type == 'aim':
            self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)
        else:
            if output_activation is None:
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.tanh)
            elif output_activation=='tanh':
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.relu, output_nonlinearity=torch.tanh, tanh_constant = tanh_constant)
            
        self.d.to(self.device)
        self.lr = lr
        if optim=='adam':
            self.discriminator_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=adam_eps)
        
        self.lipschitz_constant = lipschitz_constant 
        self.env_name = env_name
        self.lambda_coef = lambda_coef
        self.apply(utils.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        return discriminator output
        :param x:
        :return:
        """
        
        output = self.d(x)
        return output

    def reward(self, x: torch.Tensor) -> np.ndarray:
        """
        return the reward
        """
        
        r = self.forward(x)
        if self.reward_type is not None:
            r = self.reward_type(r)
        return r.cpu().detach().numpy()

    def compute_graph_pen(self,
                          prev_state: torch.Tensor,
                          next_state_state: torch.Tensor):
        """
        Computes values of the discriminator at different points
        and constraints the difference to be 0.1
        """
        if self.use_cuda:
            prev_state = prev_state.cuda()
            next_state_state = next_state_state.cuda()
            zero = torch.zeros(size=[int(next_state_state.size(0))]).cuda()
        else:
            zero = torch.zeros(size=[int(next_state_state.size(0))])
        prev_out = self(prev_state)
        next_out = self(next_state_state)
        penalty = self.lambda_coef * torch.max(torch.abs(next_out - prev_out) - self.lipschitz_constant, zero).pow(2).mean()
        return penalty


    def optimize_discriminator(self, target_states, policy_states, policy_next_states,
                                replay_buffer = None, goal_env = None,):
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        self.discriminator_optimizer.zero_grad()
        
        ones = target_states # [bs, dim([ag,dg])] #[g,g]
        zeros = policy_next_states # [bs, dim([ag,dg])] #[s',g]
        zeros_prev = policy_states # [bs, dim([ag,dg])] #[s,g]

        pred_ones = self(ones)
        pred_zeros = self(zeros)
        graph_penalty = self.compute_graph_pen(zeros_prev, zeros)
        min_aim_f_loss = None
        wgan_loss = torch.mean(pred_zeros) + torch.mean(pred_ones * (-1.))                
        loss = wgan_loss + graph_penalty

        

        loss.backward()
        self.discriminator_optimizer.step()
        return loss, wgan_loss, graph_penalty, min_aim_f_loss


class OUTPACEAgent(object):
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, encoder_target_cfg, critic_cfg, critic_target_cfg, 
                 actor_cfg, 
                 discount,
                 init_temperature, lr, actor_update_frequency,
                 critic_target_tau, critic_target_update_frequency,
                 encoder_target_tau, encoder_update_frequency, batch_size,
                 num_seed_steps,                 
                 env_name = None,                 
                 use_aim = True, aim_discriminator_cfg = None, aim_kwargs = None,
                 consider_done_true_in_critic = False,   
                 inv_weight_curriculum_kwargs = None,
                 use_meta_nml = False, meta_nml_cfg = None, meta_nml_kwargs = None,
                 normalize_nml_obs = False,
                 normalize_f_obs = False,
                 normalize_rl_obs = False,
                 randomwalk_method = 'rand_action',
                 goal_dim = None, 
                 use_aim_disc_ensemble = False,
                 adam_eps = 1e-8, optim='adam',
                 rl_reward_type = 'aim',
                 
                 ):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_frequency = critic_target_update_frequency
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency
        self.batch_size = batch_size
        self.goal_dim = goal_dim
        
        self.num_seed_steps = num_seed_steps
        self.lr = lr
        self.rl_reward_type = rl_reward_type
        
        self.encoder = encoder_cfg.to(self.device) 
        self.encoder_target = encoder_target_cfg.to(self.device) 
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        actor_cfg.repr_dim = self.encoder.repr_dim
        critic_cfg.repr_dim = self.encoder.repr_dim
        
        
        self.actor = actor_cfg.to(self.device) 
        self.critic = critic_cfg.to(self.device) 
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        

        self.alpha_lr = 1e-5
          
        self.env_name = env_name        

        self.use_aim = use_aim
        self.use_aim_disc_ensemble = use_aim_disc_ensemble        
        if use_aim:
            self.aim_discriminator = aim_discriminator_cfg.to(self.device)
            self.aim_kwargs = aim_kwargs
            self.aim_disc_update_frequency = aim_kwargs['aim_disc_update_frequency']
            self.aim_rew_std = aim_kwargs['aim_rew_std'] 
            self.aim_rew_mean = aim_kwargs['aim_rew_mean']
            self.aim_discriminator_steps = aim_kwargs['aim_discriminator_steps']
            self.aim_reward_normalize = aim_kwargs['aim_reward_normalize']
            self.aim_reward_norm_offset = aim_kwargs['aim_reward_norm_offset']
            self.aim_input_type = aim_kwargs['aim_input_type']
        else:
            self.aim_discriminator = None

        
        self.consider_done_true_in_critic = consider_done_true_in_critic
        self.adam_eps = adam_eps
        self.optim = optim
        
        
        self.initial_states = None
        self.final_goal_states = None
        self.inv_weight_curriculum_kwargs = inv_weight_curriculum_kwargs
        self.inv_weight_curriculum_temperature = inv_weight_curriculum_kwargs['inv_weight_curriculum_temperature']        
        self.inv_weight_curriculum_type = inv_weight_curriculum_kwargs['inv_weight_curriculum_type']        
        self.inv_weight_curriculum_logit_type = inv_weight_curriculum_kwargs['inv_weight_curriculum_logit_type']                    
        self.use_Vf_to_inv_curriculum = inv_weight_curriculum_kwargs['use_Vf_to_inv_curriculum']

        self.normalize_nml_obs = normalize_nml_obs
        self.normalize_f_obs = normalize_f_obs
        self.normalize_rl_obs = normalize_rl_obs
        self.randomwalk_method = randomwalk_method
        
        self.critic_target = critic_target_cfg.to(self.device) 
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.use_meta_nml = use_meta_nml
        if use_meta_nml:
            self.meta_nml_kwargs = meta_nml_kwargs
            self.meta_nml = meta_nml_cfg
            self.equal_pos_neg_test = meta_nml_kwargs['equal_pos_neg_test']
            self.meta_nml_negatives_only = meta_nml_kwargs['meta_nml_negatives_only']
            self.meta_nml_train_every_k = meta_nml_kwargs['meta_nml_train_every_k']
            self.meta_nml_train_on_positives = meta_nml_kwargs['meta_nml_train_on_positives']
            self.meta_nml_use_preprocessor = meta_nml_kwargs['meta_nml_use_preprocessor']
            self.meta_nml_custom_embedding_key = meta_nml_kwargs['meta_nml_custom_embedding_key']
            self.meta_task_batch_size = meta_nml_kwargs['meta_task_batch_size']
            self.meta_nml_shuffle_states = meta_nml_kwargs['meta_nml_shuffle_states']
            self.num_initial_meta_epochs = meta_nml_kwargs['num_initial_meta_epochs']
            self.num_meta_epochs = meta_nml_kwargs['num_meta_epochs']
            self.nml_grad_steps = meta_nml_kwargs['nml_grad_steps']
            self.test_strategy = meta_nml_kwargs['test_strategy']
            self.accumulation_steps = meta_nml_kwargs['accumulation_steps']
            
            self.meta_train_sample_size = meta_nml_kwargs['meta_train_sample_size']
            self.meta_test_sample_size = meta_nml_kwargs['meta_test_sample_size']
            self.meta_test_batch_size = meta_nml_kwargs['meta_test_batch_size']
            self.mixup_alpha = meta_nml_kwargs['mixup_alpha']             
            self.meta_nml_temperature = meta_nml_kwargs['meta_nml_temperature']
        
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]
        # optimizers
        self.init_optimizers(lr)

        self.train()
        
        self.critic_target.train()        
        self.encoder_target.train()
    
        

    def init_optimizers(self, lr):
        if self.optim=='adam':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=self.adam_eps)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=lr, eps=self.adam_eps)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr, eps=self.adam_eps) #eps=1e-02
            
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)        
        self.encoder.train(training)
        
        if self.use_aim:
            self.aim_discriminator.train(training)
        

    @property
    def alpha(self):
        return self.log_alpha.exp()
    

    def act(self, obs, spec, sample=False):
        if self.normalize_rl_obs:
            obs = self.normalize_obs(obs, self.env_name)


        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        obs = self.encoder.encode(obs)

        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])        
    

    # MetaNML related
    def sample_negatives(self, replay_buffer, goal_env, size): # from replay buffer        
        obs, _, _, _, _, _ = replay_buffer.sample_without_relabeling(size, self.discount, sample_only_state = False)            
        batch = goal_env.convert_obs_to_dict(obs.detach().cpu().numpy())['achieved_goal']
        
        negatives = batch
        labels = np.zeros(len(negatives))
        
        return negatives.astype(np.float32), labels

    def sample_positives(self, size): # from final goal
        
        final_goal = self.final_goal_states.copy()
        
        rand_positive_ind = np.random.randint(0, final_goal.shape[0], size=size)
        
        batch = final_goal[rand_positive_ind]
        
        positives = batch

        return positives.astype(np.float32), np.ones(len(positives))

    def sample_meta_test_batch(self, size, replay_buffer=None, goal_env=None):
        if self.meta_nml_negatives_only:
            return self.sample_negatives(replay_buffer, goal_env, size)
        else:
            negatives = self.sample_negatives(replay_buffer, goal_env, size // 2)
            positives = self.sample_positives(size // 2)
            return tuple(np.concatenate([a, b], axis=0) for a, b in zip(negatives, positives))
    
    
    
    def get_prob_by_meta_nml(self, observations, epoch, replay_buffer=None, goal_env=None):
        
        if epoch == 0:
            finetuning_sample = None
        else:
            finetuning_sample = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
        
        classifier_inputs = observations
        
        eval_inputs = classifier_inputs
        if self.normalize_nml_obs:
            eval_inputs = self.normalize_obs(eval_inputs, self.env_name)
            if finetuning_sample is not None:
                finetuning_sample = list(finetuning_sample)
                finetuning_sample[0] = self.normalize_obs(finetuning_sample[0], self.env_name)
                finetuning_sample = tuple(finetuning_sample)
        prob = self.meta_nml.evaluate(eval_inputs,
            num_grad_steps=self.nml_grad_steps, train_data=finetuning_sample)[:,1]
    
        return prob



    # AIM related
    def get_aim_outputs(self, obs : torch.Tensor, goal_env):
        with torch.no_grad():            
            obs_dict = goal_env.convert_obs_to_dict(obs) # torch.as_tensor(obs, device=self.device).float()
            
            if self.env_name in ['sawyer_door', 'sawyer_peg']:
                raise NotImplementedError('You should use get_object_states_only_from_goal')

            if self.aim_input_type=='default':
                obs_desired_goal = obs_dict['desired_goal']
                
                if self.inv_weight_curriculum_logit_type=='disc':
                    if self.normalize_f_obs:
                        aim_outputs = self.aim_discriminator.forward(self.normalize_obs(torch.cat([obs_dict['achieved_goal'], obs_desired_goal], dim = -1), env_name=self.env_name)).squeeze() #[bs, 1]-> [bs]
                    else:
                        aim_outputs = self.aim_discriminator.forward(torch.cat([obs_dict['achieved_goal'], obs_desired_goal], dim = -1)).squeeze() #[bs, 1]-> [bs]
                elif self.inv_weight_curriculum_logit_type =='reward':
                    aim_outputs = self.compute_aim_reward(torch.cat([obs_dict['achieved_goal'], obs_desired_goal], dim = -1)).squeeze() #[bs, 1]-> [bs]
                else:
                    raise NotImplementedError
                

        
        return aim_outputs

    def sample_idx_by_aim_outputs(self, obs : torch.Tensor, goal_env, n_sample, topk=None, return_w_prob=False):

        aim_outputs = self.get_aim_outputs(obs, goal_env)    
        
        assert topk > 1, "should be larger than 1"

        if self.inv_weight_curriculum_type=='softmin':                    
            aim_indices = torch.argsort(aim_outputs, dim = 0)[-topk:]
            aim_outputs = aim_outputs[aim_indices]
            aim_outputs_max = aim_outputs.max()
            aim_outputs_min = aim_outputs.min()
            logits = ((aim_outputs-aim_outputs_min)/(aim_outputs_max - aim_outputs_min+0.00001)-0.5)*2 #[0, 1] -> [-1,1]
            prob = F.softmin(logits/self.inv_weight_curriculum_temperature, dim = 0) #[bs]
            
            dist = torch.distributions.Categorical(probs=prob)
            sample_idx = dist.sample((n_sample,))            
            sample_idx = aim_indices[sample_idx]

            
        elif self.inv_weight_curriculum_type=='topk':
            aim_indices = torch.argsort(aim_outputs, dim = 0)[-topk:]
            sample_idx = torch.randint(0, aim_indices.shape[0], size=(n_sample,))
            sample_idx = aim_indices[sample_idx]

        elif self.inv_weight_curriculum_type=='p^-1':
            raise NotImplementedError
        elif self.inv_weight_curriculum_type=='p^-2':
            raise NotImplementedError

        

        if return_w_prob:
            sample_idx, prob
        else:
            return sample_idx # torch.from_numpy(obs).float().to(self.device)[sample_idx] if type(obs)==np.np.ndarray else obs[sample_idx]
        
    
    def normalize_obs(self, obs, env_name): 
        # normalize to [-1,1]
        if obs is None:
            return None
        if type(obs)==np.ndarray:
            obs = obs.copy()    
        elif type(obs)==torch.Tensor:
            obs = copy.deepcopy(obs)
        else:
            raise NotImplementedError

        if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            center, scale = 4, 6
        elif env_name == "PointSpiralMaze-v0":
            center, scale = 0, 10
        elif env_name in ["PointNMaze-v0"]:
            if obs.shape[-1]==self.goal_dim:
                center, scale = np.array([4, 8]), np.array([6, 10])
                if torch.is_tensor(obs):
                    center = torch.from_numpy(center).to(self.device)
                    scale = torch.from_numpy(scale).to(self.device)
                obs = (obs-center)/scale
            elif obs.shape[-1]==self.goal_dim*2:
                center, scale = np.array([4, 8, 4, 8]), np.array([6, 10, 6, 10])
                if torch.is_tensor(obs):
                    center = torch.from_numpy(center).to(self.device)
                    scale = torch.from_numpy(scale).to(self.device)
                obs = (obs-center)/scale
            else:
                center, scale = np.array([4, 8]), np.array([6, 10])
                if torch.is_tensor(obs):
                    center = torch.from_numpy(center).to(self.device)
                    scale = torch.from_numpy(scale).to(self.device)
                obs[..., :2] = (obs[..., :2]-center)/scale
                center, scale = np.array([4, 8, 4, 8]), np.array([6, 10, 6, 10])
                if torch.is_tensor(obs):
                    center = torch.from_numpy(center).to(self.device)
                    scale = torch.from_numpy(scale).to(self.device)
                obs[..., -4:] = (obs[..., -4:]-center)/scale
            return obs
        elif env_name in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
            # normalization maybe not needed
            assert self.normalize_nml_obs, 'should normalize nml obs as nml can distinguish in large scale'
            assert not self.normalize_f_obs, 'should not normalize f obs as as scale is 0.05 for normlaize_nml_obs'
            assert not self.normalize_rl_obs, 'should not normalize rl obs as as scale is 0.05 for normlaize_nml_obs'
            center, scale = 0, 0.05
        else:
            raise NotImplementedError

        if obs.shape[-1]==self.goal_dim or obs.shape[-1]==self.goal_dim*2:
            obs = (obs-center)/scale
        else:
            obs[..., :2] = (obs[..., :2]-center)/scale
            obs[..., -4:] = (obs[..., -4:]-center)/scale

        return obs
        
    
    
    def sample_randomwalk_goals(self, obs, ag, episode, env, replay_buffer, num_candidate = 5, random_noise = 2.5, uncertainty_mode = 'f'):
        noise = np.random.uniform(low=-random_noise, high=random_noise, size=(num_candidate, env.goal_dim))

        if self.env_name in ['sawyer_peg_pick_and_place']:
            pass
        elif self.env_name in ['sawyer_peg_push']:
            noise[2] = 0
            
        candidate_goal = np.tile(ag, (num_candidate,1)) + noise
        
        if uncertainty_mode == 'nml' and self.use_meta_nml:
            start = time.time()
            classification_probabilities = self.get_prob_by_meta_nml(candidate_goal, episode, replay_buffer=replay_buffer, goal_env=env)
            # print('get prob by meta nml time in sample_randomwalk_goals : ', time.time() - start)

            satisfied = False
            epsilon = 0
            while not satisfied:                
                lb = 0.4-epsilon                
                if lb <0:
                    warnings.warn('meta_nml uncertainty threshold is out of range!!')
                
                uncertain_indices = np.where(((classification_probabilities>=0.4-epsilon))==1)[0]
                if uncertain_indices.shape[0]==0:
                    epsilon +=0.02
                else:
                    satisfied = True
        
            prob = F.softmax(torch.from_numpy(classification_probabilities[uncertain_indices]/self.meta_nml_temperature).float().to(self.device), dim = 0)
            dist = torch.distributions.Categorical(probs=prob)
            idxs = dist.sample((1,)).detach().cpu().numpy()
            

            obs = candidate_goal[uncertain_indices[idxs]]
            
        elif uncertainty_mode == 'f':
            aim_obs = np.tile(obs, (candidate_goal.shape[0], 1))
            aim_obs[:, -env.goal_dim:] = candidate_goal
            indices = self.sample_idx_by_aim_outputs(torch.as_tensor(aim_obs, device=self.device).float(), env, 1, topk=aim_obs.shape[0])
            obs = aim_obs[indices][-env.goal_dim:]

        
        return np.squeeze(obs)


    def update_critic(self, obs, action, reward, next_obs, discount, done, step):
        offline_rl_dict = None
        discor_dict = None                        

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        
            
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                target_Q2) - self.alpha.detach() * log_prob
        
            # target_Q = reward + (discount * target_V)
            if self.consider_done_true_in_critic:
                # print('consider done true in critic')
                target_Q = reward + (discount * target_V)*(1-done)
            else:
                # print('do not consider done true in critic')
                target_Q = reward + (discount * target_V)

        
        # get current Q estimates
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        return Q1, Q2, critic_loss, offline_rl_dict, discor_dict
    
    
    def sparse_reward(self, ag, dg, threshold = 0.5):
        if(ag.dim() == 1):
            dist = torch.linalg.norm(ag - dg)
        else:
            dist = torch.linalg.norm(ag - dg, axis=1)
            
        rs = (dist > threshold)
        return torch.unsqueeze(-rs.type(torch.float32), dim=1)
    

    def update_actor_and_alpha(self, obs, step, offline_rl_dict = None):        
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # use custom alpha update 
        custom_alpha_optimize = True
        if custom_alpha_optimize:
            alpha_loss = (self.alpha*(-log_prob - self.target_entropy).detach()).mean() # just for logging
            alpha_loss_grad = (-log_prob - self.target_entropy).detach().mean()
            alpha = torch.clamp(self.alpha - self.alpha_lr*alpha_loss_grad , min=0.001, max = 0.5)
            self.log_alpha = alpha.log()

        else:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, log_prob
        
   
    def update_aim_discriminator(self, aim_disc_replay_buffer, goal_env):
        obs, action, extr_reward, next_obs, discount, dones = aim_disc_replay_buffer.sample(
            self.batch_size, self.discount)
        obs_dict, next_obs_dict = map(goal_env.convert_obs_to_dict, (obs, next_obs))
        
        if self.env_name in ['sawyer_door', 'sawyer_peg']: 
            scale =0.01
            if self.aim_input_type=='default':                
                # To obtain only object related states for AIM
                policy_states = torch.cat([get_object_states_only_from_goal(self.env_name, obs_dict['achieved_goal']), get_object_states_only_from_goal(self.env_name, obs_dict['desired_goal'])], dim = -1) # s, s_g
                policy_next_states = torch.cat([get_object_states_only_from_goal(self.env_name, next_obs_dict['achieved_goal']), get_object_states_only_from_goal(self.env_name, next_obs_dict['desired_goal'])], dim = -1) # s', s_g            
                target_states = torch.cat([get_object_states_only_from_goal(self.env_name, next_obs_dict['desired_goal']) + torch.from_numpy(np.random.normal(scale=scale, size=get_object_states_only_from_goal(self.env_name, next_obs_dict['desired_goal']).shape)).float().to(self.device), get_object_states_only_from_goal(self.env_name, next_obs_dict['desired_goal'])], dim = -1) # s_g, s_g

        else:            
            obs_desired_goal = obs_dict['desired_goal']
            next_obs_desired_goal = next_obs_dict['desired_goal']
            if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0', 'PointNMaze-v0', 'PointSpiralMaze-v0']:
                scale = 0.1
            elif self.env_name in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
                scale = 0.01
            else:
                raise NotImplementedError



            if self.aim_input_type=='default':
                policy_states = torch.cat([obs_dict['achieved_goal'], obs_desired_goal], dim = -1) # s, s_g
                policy_next_states = torch.cat([next_obs_dict['achieved_goal'], next_obs_desired_goal], dim = -1) # s', s_g            
                target_states = torch.cat([next_obs_desired_goal+ torch.from_numpy(np.random.normal(scale=scale, size=next_obs_desired_goal.shape)).float().to(self.device), next_obs_desired_goal], dim = -1) # s_g, s_g
            
        
        if self.normalize_f_obs:
            target_states = self.normalize_obs(target_states, self.env_name)
            policy_states = self.normalize_obs(policy_states, self.env_name)
            policy_next_states = self.normalize_obs(policy_next_states, self.env_name)
        
        self.aim_disc_loss, self.wgan_loss, self.graph_penalty, self.min_aim_f_loss = \
            self.aim_discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)
            

        all_rewards = []
        all_rewards.append(self.aim_discriminator.reward(target_states))
        all_rewards.append(self.aim_discriminator.reward(policy_states))
        return self.aim_disc_loss, all_rewards

    def compute_aim_reward(self, obs):
        if self.normalize_f_obs:
            obs = self.normalize_obs(obs, self.env_name)
        aim_reward = self.aim_discriminator.forward(obs)
        if self.aim_reward_normalize:            
            aim_reward = (aim_reward - self.aim_rew_mean)/(self.aim_rew_std*2.)
        return aim_reward
    
    

    def update(self, replay_buffer, randomwalk_buffer, aim_disc_replay_buffer, step, goal_env = None, goal_buffer = None):
        
        if self.use_aim:
            if step == self.num_seed_steps // 2:
                drewards = []
                for disc_step in range(self.aim_discriminator_steps):
                    _, rsamples = self.update_aim_discriminator(aim_disc_replay_buffer, goal_env)                
                    drewards.extend(rsamples)
                drewards = np.reshape(drewards, newshape=(-1,))
                self.aim_rew_std = np.std(drewards) + self.aim_reward_norm_offset # 0.1
                self.aim_rew_mean = np.max(drewards) + self.aim_reward_norm_offset # 0.1


                if self.use_meta_nml:
                    num_epochs = self.num_initial_meta_epochs
                    
                    if self.meta_nml_train_on_positives:
                        train_data = self.sample_meta_test_batch(self.meta_train_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                    else:
                        train_data = self.sample_negatives(replay_buffer, goal_env, self.meta_train_sample_size)
                    test_data = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                    meta_train_start = time.time()
                    if self.normalize_nml_obs:
                        train_data = list(train_data)
                        test_data = list(test_data)
                        train_data[0] = self.normalize_obs(train_data[0], self.env_name)
                        test_data[0] = self.normalize_obs(test_data[0], self.env_name)
                        train_data = tuple(train_data)
                        test_data = tuple(test_data)
                    self.meta_nml_results = self.meta_nml.train(train_data, test_data, 
                                        batch_size=self.meta_task_batch_size, accumulation_steps=self.accumulation_steps,
                                        num_epochs=num_epochs, test_strategy=self.test_strategy, 
                                        test_batch_size=self.meta_test_batch_size, mixup_alpha=self.mixup_alpha)
                    self.meta_train_time = (time.time() - meta_train_start)
                

        else:
            if step == self.num_seed_steps // 2:

                if self.use_meta_nml:
                    num_epochs = self.num_initial_meta_epochs
                    
                    if self.meta_nml_train_on_positives:
                        train_data = self.sample_meta_test_batch(self.meta_train_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                    else:
                        train_data = self.sample_negatives(replay_buffer, goal_env, self.meta_train_sample_size)
                    test_data = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                    meta_train_start = time.time()
                    if self.normalize_nml_obs:
                        train_data = list(train_data)
                        test_data = list(test_data)
                        train_data[0] = self.normalize_obs(train_data[0], self.env_name)
                        test_data[0] = self.normalize_obs(test_data[0], self.env_name)
                        train_data = tuple(train_data)
                        test_data = tuple(test_data)
                    self.meta_nml_results = self.meta_nml.train(train_data, test_data, 
                                        batch_size=self.meta_task_batch_size, accumulation_steps=self.accumulation_steps,
                                        num_epochs=num_epochs, test_strategy=self.test_strategy, 
                                        test_batch_size=self.meta_test_batch_size, mixup_alpha=self.mixup_alpha)
                    self.meta_train_time = (time.time() - meta_train_start)
          

        # if len(replay_buffer) < self.num_seed_steps:
        if step < self.num_seed_steps:
            return
        if self.use_aim:
            if step % self.aim_disc_update_frequency == 0 :
                # assume goal concatenated obs
                drewards = []
                for disc_step in range(self.aim_discriminator_steps):
                    _, rsamples = self.update_aim_discriminator(aim_disc_replay_buffer, goal_env)                
                    drewards.extend(rsamples)
                drewards = np.reshape(drewards, newshape=(-1,))
                self.aim_rew_std = np.std(drewards) + self.aim_reward_norm_offset # 0.1
                self.aim_rew_mean = np.max(drewards) + self.aim_reward_norm_offset # 0.1
            
                        
            if self.use_meta_nml and step % self.meta_nml_train_every_k == 0 :
                num_epochs = self.num_meta_epochs        
                
                if self.meta_nml_train_on_positives:
                    train_data = self.sample_meta_test_batch(self.meta_train_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                else:
                    train_data = self.sample_negatives(replay_buffer, goal_env, self.meta_train_sample_size)
                test_data = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                meta_train_start = time.time()
                if self.normalize_nml_obs:
                    train_data = list(train_data)
                    test_data = list(test_data)
                    train_data[0] = self.normalize_obs(train_data[0], self.env_name)
                    test_data[0] = self.normalize_obs(test_data[0], self.env_name)
                    train_data = tuple(train_data)
                    test_data = tuple(test_data)
                self.meta_nml_results = self.meta_nml.train(train_data, test_data, 
                                    batch_size=self.meta_task_batch_size, accumulation_steps=self.accumulation_steps,
                                    num_epochs=num_epochs, test_strategy=self.test_strategy, 
                                    test_batch_size=self.meta_test_batch_size, mixup_alpha=self.mixup_alpha)
                self.meta_train_time = (time.time() - meta_train_start)

        else:            
            if self.use_meta_nml and step % self.meta_nml_train_every_k == 0 :
                num_epochs = self.num_meta_epochs        
                
                if self.meta_nml_train_on_positives:
                    train_data = self.sample_meta_test_batch(self.meta_train_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                else:
                    train_data = self.sample_negatives(replay_buffer, goal_env, self.meta_train_sample_size)
                test_data = self.sample_meta_test_batch(self.meta_test_sample_size, replay_buffer=replay_buffer, goal_env=goal_env)
                meta_train_start = time.time()
                if self.normalize_nml_obs:
                    train_data = list(train_data)
                    test_data = list(test_data)
                    train_data[0] = self.normalize_obs(train_data[0], self.env_name)
                    test_data[0] = self.normalize_obs(test_data[0], self.env_name)
                    train_data = tuple(train_data)
                    test_data = tuple(test_data)
                self.meta_nml_results = self.meta_nml.train(train_data, test_data, 
                                    batch_size=self.meta_task_batch_size, accumulation_steps=self.accumulation_steps,
                                    num_epochs=num_epochs, test_strategy=self.test_strategy, 
                                    test_batch_size=self.meta_test_batch_size, mixup_alpha=self.mixup_alpha)
                self.meta_train_time = (time.time() - meta_train_start)
            

        
        if randomwalk_buffer is None or self.randomwalk_method == 'rand_action':
            obs, action, extr_reward, next_obs, discount, dones = replay_buffer.sample(self.batch_size, self.discount)            
        else:
            obs, action, extr_reward, next_obs, discount, dones = utils.sample_mixed_buffer(replay_buffer, randomwalk_buffer, self.batch_size, self.discount)
        


        ###############
        if self.use_aim:
            with torch.no_grad():                
                obs_dict, next_obs_dict = map(goal_env.convert_obs_to_dict, (obs, next_obs))                
                
                next_obs_desired_goal = next_obs_dict['desired_goal']

                if self.env_name in ['sawyer_door', 'sawyer_peg']:
                    if self.aim_input_type=='default':
                        self.intr_reward = self.compute_aim_reward(torch.cat([get_object_states_only_from_goal(self.env_name, next_obs_dict['achieved_goal']), get_object_states_only_from_goal(self.env_name, next_obs_desired_goal)], dim = -1))
                    
                else:
                    if self.aim_input_type=='default':
                        self.intr_reward = self.compute_aim_reward(torch.cat([next_obs_dict['achieved_goal'], next_obs_desired_goal], dim = -1))                
                    
            if self.rl_reward_type == "aim":
                reward = self.intr_reward
            elif self.rl_reward_type == "sparse":
                obs_dict, next_obs_dict = map(goal_env.convert_obs_to_dict, (obs, next_obs))  
            
                next_obs_desired_goal = next_obs_dict['desired_goal']
                next_obs_achieved_goal = next_obs_dict['achieved_goal']        
                if self.env_name in ['AntMazeSmall-v0']:
                    threshold = 1.0
                elif self.env_name in ['PointUMaze-v0', "PointSpiralMaze-v0", "PointNMaze-v0"]:
                    threshold = 0.5
                elif self.env_name in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
                    threshold = 0.05
                else:
                    raise NotImplementedError

                reward = self.sparse_reward(next_obs_achieved_goal, next_obs_desired_goal, threshold =threshold)
            else:
                raise NotImplementedError
                
        else:
            obs_dict, next_obs_dict = map(goal_env.convert_obs_to_dict, (obs, next_obs))  
            
            next_obs_desired_goal = next_obs_dict['desired_goal']
            next_obs_achieved_goal = next_obs_dict['achieved_goal']        
            if self.env_name in ['AntMazeSmall-v0']:
                threshold = 1.0
            elif self.env_name in ['PointUMaze-v0', "PointSpiralMaze-v0", "PointNMaze-v0"]:
                threshold = 0.5
            elif self.env_name in ['sawyer_peg_push','sawyer_peg_pick_and_place']:
                threshold = 0.05
            else:
                raise NotImplementedError
                
            reward = self.sparse_reward(next_obs_achieved_goal, next_obs_desired_goal, threshold =threshold)
        
        
        
        # From here, RL related 
        if self.normalize_rl_obs:
            obs = self.normalize_obs(obs, self.env_name)
            next_obs = self.normalize_obs(next_obs, self.env_name)

        # decouple representation
        with torch.no_grad():
            obs = self.encoder.encode(obs)
            next_obs = self.encoder.encode(next_obs)
        
        Q1, Q2, critic_loss, offline_rl_dict, discor_dict = self.update_critic(obs, action, reward, next_obs, discount, dones, step)
        

        if step % self.actor_update_frequency == 0:
            self.actor_loss, self.alpha_loss, self.actor_log_prob = self.update_actor_and_alpha(obs, step, offline_rl_dict=offline_rl_dict)
            
        if step % self.critic_target_update_frequency == 0:            
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)
        
            
        # logging
        logging_dict = dict(q1=Q1.detach().cpu().numpy().mean(),
                            q2=Q2.detach().cpu().numpy().mean(),
                            critic_loss=critic_loss.detach().cpu().numpy(),
                            actor_loss = self.actor_loss.detach().cpu().numpy(),                            
                            batch_reward_mean = reward.detach().cpu().numpy().mean(),                            
                            )
        
        
        logging_dict.update(dict(
                                alpha_loss = self.alpha_loss.detach().cpu().numpy(),
                                bacth_actor_log_prob = self.actor_log_prob.detach().cpu().numpy().mean(),
                                alpha = self.alpha.detach().cpu().numpy(),
                                entropy_diff = (-self.actor_log_prob-self.target_entropy).detach().cpu().numpy().mean(),
                                ))

        
        if self.use_aim:
            logging_dict.update({'aim_disc_loss' : self.aim_disc_loss.detach().cpu().numpy()})
            logging_dict.update({'aim_intr_reward_mean' : self.intr_reward.detach().cpu().numpy().mean()})
            logging_dict.update({'aim_intr_reward_max' : self.intr_reward.detach().cpu().numpy().max()})
            logging_dict.update({'aim_intr_reward_min' : self.intr_reward.detach().cpu().numpy().min()})
            logging_dict.update({'aim_intr_reward_std' : self.intr_reward.detach().cpu().numpy().std()})
            logging_dict.update({'aim_wgan_loss' : self.wgan_loss.detach().cpu().numpy()}) # it should be decreases as it converges?
            logging_dict.update({'aim_graph_penalty_loss' : self.graph_penalty.detach().cpu().numpy()}) # it should be near zero to satisfy lipschitz constraint?


        if self.use_meta_nml: # list of dictionary
            logging_dict.update({'meta_nml_time' : self.meta_train_time})            
            all_keys = self.meta_nml_results[0].keys()
            for key in all_keys:                
                val = np.stack([res[key] for res in self.meta_nml_results], axis =0).mean()            
                logging_dict.update({'meta_nml_'+str(key) : val})


        return logging_dict

        
        
        