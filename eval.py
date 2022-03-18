import numpy as np
import cv2

import torch

from mlagents_envs.base_env import ActionTuple

from agents import RNDAgent
from utils import make_env
from config import *


if __name__ == '__main__':
    # log config file
    config_ = default_config
    settings = dict(config_.items())

    WANDB = default_config.getboolean('LogData')
    use_cuda = default_config.getboolean('UseGPU')

    # unity connection
    env, spec, behavior_name, n_envs, obs_space, action_space, len_action_space = make_env()

    settings['BatchSize'] = int(int(settings['RolloutStep']) * n_envs / int(settings['MiniBatch']))
    settings['InputDim'] = [obs_space[2], obs_space[0], obs_space[1]]
    settings['InputDimRND'] = [obs_space[2], obs_space[0], obs_space[1]]
    settings['Actions'] = action_space
    settings['UseCuda'] = use_cuda
    settings['WandB'] = WANDB
    settings['Envs'] = n_envs

    # global variables
    roll_steps = int(settings['RolloutStep'])
    obs_gray_stack = int(settings['InputDim'][0]) > 3

    if obs_gray_stack:
        settings['InputDimRND'][0] = 1

    # get config file
    print('config: \n', {f'{k}: {v}' for k, v in zip(settings.keys(), settings.values())}, '\n')

    # instantiate agent and normalizers
    agent = RNDAgent(settings)

    model_id = int(settings['ModelID'])

    print('Loading models...')
    agent.model.load_state_dict(torch.load(f'models/agent_{model_id}.pt'))
    agent.rnd.predictor.load_state_dict(torch.load(f'models/predictor_{model_id}.pt'))
    agent.rnd.target.load_state_dict(torch.load(f'models/target_{model_id}.pt'))
    print('...end load.\n')

    agent.model.eval()
    agent.rnd.predictor.eval()
    agent.rnd.target.eval()

    states = np.zeros([n_envs, *settings['InputDim']])
    while True:
        a, ve, vi, p = agent.get_action(states)

        action = ActionTuple(discrete=np.expand_dims(a, -1))
        env.set_actions(behavior_name, action)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        states_ = np.transpose(np.array(decision_steps.obs), (0, 1, 4, 2, 3)).reshape(n_envs, *settings['InputDim'])

        # display agent's state
        if obs_gray_stack:
            cv2.imshow('state', np.array(decision_steps.obs[0][0][:, :, 0]))
            cv2.waitKey(1)
        else:
            cv2.imshow('state', cv2.cvtColor(np.array(decision_steps.obs[0][0][:, :, :]), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        states = states_

