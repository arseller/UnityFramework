import numpy as np
import torch

from torch._six import inf
from config import *

from mlagents_envs.environment import UnityEnvironment


use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])


def make_env():
    """
    Connection to Unity Environment

    :return: env, behavior_name, n_envs, obs_space, action_space
    """
    print('\nConnecting to Unity Environment...\n')
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
    env.reset()
    print('--- Environment Specs ---')

    behavior_name = list(env.behavior_specs)[0]

    # Env specs
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    print('Behavior name :', behavior_name)
    n_envs = len(decision_steps)
    print('Number of Environment : ', n_envs)
    obs_space = list(decision_steps.obs[0][0, :, :, :].shape)
    print('Observation space :', obs_space)

    # Behavior specs
    spec = env.behavior_specs[behavior_name]

    # Action
    if spec.action_spec.discrete_size > 0:
        for branch, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f'Action space : {branch + 1} discrete branch with {branch_size} different options')
    action_space = list(spec.action_spec.discrete_branches)
    len_action_space = len(decision_steps)
    print()

    return env, spec, behavior_name, n_envs, obs_space, action_space[0], len_action_space


def compute_target_adv(reward, done, value, gamma, n_step, n_envs):
    discounted_return = np.empty([n_envs, n_step])
    gae = np.zeros_like([n_envs, ])
    for t in range(n_step - 1, -1, -1):
        delta = reward[t] + gamma * value[t + 1] * (1 - done[t]) - value[t]
        gae = delta + gamma * lam * (1 - done[t]) * gae
        discounted_return[:, t] = gae + value[t]
    adv = discounted_return - value[:-1].T

    return discounted_return.reshape([-1]), adv.reshape([-1])


def global_grad_norm(parameters, norm_type=2):
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.

    :param parameters: (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have
            gradients normalized
    :param norm_type: (float or int): type of the used p-norm. Can be 'inf' for infinity norm.

    :return: Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def check_grad(params):
    for name, net in params:
        if net.grad is None:
            print('Gradients None -> Name', name)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

