import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, project_for_state_input = False):
        super().__init__()
        
        assert len(obs_shape) == 1
        
        self.repr_dim = obs_shape[-1]
        # only for matching the outpacetype's dim
        self.project_for_state_input = project_for_state_input
        if project_for_state_input:
            self.projector = nn.Linear(self.repr_dim, self.repr_dim)

    def encode(self, obs):
        return obs

    def forward(self, obs):
        h = self.encode(obs)
        if self.project_for_state_input:
            z = self.projector(h)
        else:
            z = h
            
        return z

class StateActor(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim,
                 hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        
        self.fc = utils.mlp(feature_dim, hidden_dim, 2 * action_shape[0],
                            hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs):
        
        h = obs
        mu, log_std = self.fc(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist

class StateCritic(nn.Module):
    def __init__(self, feature_dim, action_shape, hidden_dim,
                 hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                            hidden_depth)
        self.Q2 = utils.mlp(feature_dim + action_shape[0], hidden_dim, 1,
                            hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        h = obs
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


