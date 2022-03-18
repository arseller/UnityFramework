from abc import ABC

import numpy as np

import torch
import torch.nn as nn

from torch.nn import init


class PPOModel(nn.Module, ABC):
    def __init__(self, n_envs, input_dim, actions):
        super(PPOModel, self).__init__()

        # env: [n_envs]
        # input_dim: [channel, height, width]
        # output_dim: [actions]

        self.batch_dim = n_envs
        self.in_channels = input_dim[0]
        self.input_dim = input_dim
        self.actions = actions

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv_shape = self.compute_conv_shape((self.batch_dim, *self.input_dim))
        self.linear_common1 = nn.Linear(self.conv_shape, 512)
        self.linear_common2 = nn.Linear(512, 512)
        self.linear_actor1 = nn.Linear(512, 256)
        self.linear_actor2 = nn.Linear(256, self.actions)
        self.linear_common_critic = nn.Linear(512, 512)

        self.features = nn.Sequential(
            self.conv1, nn.ELU(),
            self.conv2, nn.ELU(),
            self.conv3, nn.ELU()
        )

        self.common_linear = nn.Sequential(
            self.linear_common1, nn.ELU(),
            self.linear_common2, nn.ELU(),
        )

        self.actor = nn.Sequential(
            self.linear_actor1, nn.ELU(),
            self.linear_actor2
        )

        self.common_critic_layer = nn.Sequential(
            self.linear_common_critic, nn.ELU()
        )

        self.ext_critic = nn.Linear(512, 1)
        self.int_critic = nn.Linear(512, 1)

        # weights initialization
        # conv2d
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        # recurrent linear
        init.orthogonal_(self.linear_common1.weight, np.sqrt(2))
        self.linear_common1.bias.data.zero_()
        init.orthogonal_(self.linear_common2.weight, np.sqrt(2))
        self.linear_common2.bias.data.zero_()
        # actor
        init.orthogonal_(self.linear_actor1.weight, 0.01)
        self.linear_actor1.bias.data.zero_()
        init.orthogonal_(self.linear_actor2.weight, 0.01)
        self.linear_actor2.bias.data.zero_()
        # common critic
        init.orthogonal_(self.linear_common_critic.weight, 0.01)
        self.linear_common_critic.bias.data.zero_()
        # ext critic
        init.orthogonal_(self.ext_critic.weight, 0.01)
        self.ext_critic.bias.data.zero_()
        # int critic
        init.orthogonal_(self.int_critic.weight, 0.01)
        self.int_critic.bias.data.zero_()

    def forward(self, state: torch.Tensor):
        features = self.features(state)
        flat = features.reshape((-1, self.conv_shape))
        x = self.common_linear(flat)
        policy = self.actor(x)
        value_ext = self.ext_critic(self.linear_common_critic(x) + x)
        value_int = self.int_critic(self.linear_common_critic(x) + x)
        output_data = {'policy': policy, 'value_ext': value_ext, 'value_int': value_int}
        return output_data

    def compute_conv_shape(self, input_dims):
        tmp = torch.zeros(*input_dims)
        dim = self.conv1(tmp)
        dim = self.conv2(dim)
        dim = self.conv3(dim)
        return int(np.prod(dim.shape[1:]))

    def init_hidden(self):
        h = torch.zeros((self.batch_dim, 256), dtype=torch.float32)
        return h


class Flatten(nn.Module, ABC):
    def forward(self, input_dim):
        return input_dim.reshape(input_dim.size(0), -1)


class RNDModel(nn.Module, ABC):
    def __init__(self, envs, input_dim_rnd, encoding_size=512):
        super(RNDModel, self).__init__()

        self.input_dim_rnd = input_dim_rnd
        self.in_channels = input_dim_rnd[0]
        self.encoding_size = encoding_size
        self.envs = envs

        feature_output = 7 * 7 * 64

        # target network
        self.target = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, self.encoding_size)
        )

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self.encoding_size)
        )

        # weights initialization
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs: torch.Tensor):
        predictor_feature = self.predictor(next_obs)
        target_feature = self.target(next_obs)
        output_data = {'predictor_features': predictor_feature, 'target_features': target_feature}
        return output_data

