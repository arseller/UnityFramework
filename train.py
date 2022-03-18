import wandb
import time
import cv2

import numpy as np
import torch

from mlagents_envs.base_env import ActionTuple

from agents import RNDAgent
from utils import make_env, compute_target_adv, RunningMeanStd, RewardForwardFilter
from config import *

# TODO: add remind

if __name__ == '__main__':
    np.random.seed(71)
    torch.manual_seed(71)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # log config file
    config_ = default_config
    settings = dict(config_.items())

    # config bool
    WANDB = default_config.getboolean('LogData')
    use_cuda = default_config.getboolean('UseGPU')
    render = default_config.getboolean('Render')
    hot_reload = default_config.getboolean('HotReload')
    obs_norm = default_config.getboolean('ObsNormalization')
    rew_norm = default_config.getboolean('RewNormalization')

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
    model_id = int(settings['ModelID'])
    name = 'curiosity_' + settings['TrainMethod']
    obs_gray_stack = int(settings['InputDim'][0]) > 3

    if obs_gray_stack:
        settings['InputDimRND'][0] = 1

    # get config file
    print('config: \n', {f'{k}: {v}' for k, v in zip(settings.keys(), settings.values())}, '\n')

    # instantiate agent and normalizers
    agent = RNDAgent(settings)

    if hot_reload:
        print('Hot Reload...')
        agent.model.load_state_dict(torch.load(f'models/agent_{model_id}.pt'))
        agent.rnd.predictor.load_state_dict(torch.load(f'models/predictor_{model_id}.pt'))
        agent.rnd.target.load_state_dict(torch.load(f'models/target_{model_id}.pt'))
        print('...end loading.\n')

    observation_rms = RunningMeanStd(shape=(1, *settings['InputDimRND']))
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(gamma=float(settings['Gamma']))

    # init wandb log
    if WANDB:
        wandb.init(project='curiosity-learning', name=name, config=settings)
        wandb.watch(agent.model)
        wandb.watch(agent.rnd)

    # normalize observations
    if obs_norm:
        print('\nStart observation normalization...')
        next_obs = []
        rms_update = 0
        for step in range(roll_steps * int(settings['ObsNormStep'])):

            action = spec.action_spec.random_action(len_action_space)
            env.set_actions(behavior_name, action)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            obs = np.transpose(np.array(decision_steps.obs), (0, 1, 4, 2, 3)).reshape(n_envs, *settings['InputDim'])

            obs = obs[:, -1, :, :].reshape(n_envs, *settings['InputDimRND']) if obs_gray_stack else obs

            [next_obs.append(obs[i]) for i in range(len(obs))]

            if len(next_obs) % (roll_steps * int(settings['ObsNormStep'])) == 0:
                rms_update += 1
                next_obs = np.stack(next_obs)
                observation_rms.update(next_obs)
                print('running-mean-std update {}/{}'.format(rms_update, int(settings['ObsNormStep'])))
                next_obs = []
        print('...end observation normalization.\n')

    global_update = 0
    global_step = 0
    model_id = 0

    states = np.zeros([n_envs, *settings['InputDim']])

    # start training
    start = time.time()
    while True:
        print(f'\nGlobal update: {global_update}')
        print(f'Environment steps: {global_step}')
        print('Elapsed time: {:.2f}min'.format((time.time() - start) / 60.))
        total_states, total_rewards, total_dones, total_states_, total_actions, total_int_rewards, \
        total_ext_values, total_int_values, total_policies_np, total_next_obs, total_policies = \
            [], [], [], [], [], [], [], [], [], [], torch.empty(0).to(agent.device)

        global_step += (n_envs * roll_steps)
        global_update += 1

        # 1. n-step rollout
        for _ in range(roll_steps):
            a, ve, vi, p = agent.get_action(states)

            action = ActionTuple(discrete=np.expand_dims(a, -1))
            env.set_actions(behavior_name, action)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # display agent's state
            if render:
                if obs_gray_stack:
                    cv2.imshow('state', np.array(decision_steps.obs[0][0][:, :, 0]))
                    cv2.waitKey(1)
                else:
                    cv2.imshow('state', cv2.cvtColor(np.array(decision_steps.obs[0][0][:, :, :]), cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # unity observations ∈ [0, 1]
            states_ = np.transpose(np.array(decision_steps.obs), (0, 1, 4, 2, 3)).reshape(n_envs, *settings['InputDim'])
            rewards = np.array(decision_steps.reward)
            dones = np.array([decision_steps.agent_id[i] in terminal_steps for i in range(n_envs)])

            if obs_norm:
                if obs_gray_stack:
                    next_obs = states_[:, -1, :, :].reshape(n_envs, *settings['InputDimRND'])
                    next_state = ((next_obs - observation_rms.mean) / np.sqrt(observation_rms.var)).clip(-5, 5)
                else:
                    next_state = ((states_ - observation_rms.mean) / np.sqrt(observation_rms.var)).clip(-5, 5)
            else:
                if obs_gray_stack:
                    next_obs = states_[:, -1, :, :].reshape(n_envs, *settings['InputDimRND'])
                    next_state = next_obs
                else:
                    next_state = states_

            # compute intrinsic_rewards
            intrinsic_rewards = agent.compute_intrinsic_reward(next_state)

            total_actions = np.append(total_actions, a).reshape(-1, n_envs)
            total_ext_values = np.append(total_ext_values, ve).reshape(-1, n_envs)
            total_int_values = np.append(total_int_values, vi).reshape(-1, n_envs)
            total_policies = torch.cat((total_policies, p))
            total_policies_np = np.append(total_policies_np, p.detach().cpu().numpy()).reshape(-1, n_envs, action_space)
            total_states = np.append(total_states, states).reshape(-1, n_envs, *settings['InputDim'])
            total_rewards = np.append(total_rewards, rewards).clip(-1, 1).reshape(-1, n_envs)
            total_int_rewards = np.append(total_int_rewards, intrinsic_rewards).reshape(-1, n_envs)
            total_dones = np.append(total_dones, dones).reshape(-1, n_envs)
            total_states_ = np.append(total_states_, states_).reshape(-1, n_envs, *settings['InputDim'])
            if obs_gray_stack:
                total_next_obs = np.append(total_next_obs, next_obs).reshape(-1, n_envs, *settings['InputDimRND'])

            states = states_

        # compute last next value
        _, ve, vi, _ = agent.get_action(states)
        total_ext_values = np.append(total_ext_values, ve).reshape(-1, n_envs)
        total_int_values = np.append(total_int_values, vi).reshape(-1, n_envs)
        # reshape policies' tensor
        total_policies = total_policies.reshape(-1, n_envs, action_space)
        # ------------------------------------

        assert total_actions.shape == (roll_steps, n_envs)
        assert total_ext_values.shape == ((roll_steps + 1), n_envs)
        assert total_int_values.shape == ((roll_steps + 1), n_envs)
        assert total_policies.shape == (roll_steps, n_envs, action_space)
        assert total_policies_np.shape == (roll_steps, n_envs, action_space)
        assert total_states.shape == (roll_steps, n_envs, *settings['InputDim'])
        assert total_rewards.shape == (roll_steps, n_envs,)
        assert total_int_rewards.shape == (roll_steps, n_envs,)
        assert total_dones.shape == (roll_steps, n_envs,)
        assert total_states_.shape == (roll_steps, n_envs, *settings['InputDim'])
        if obs_gray_stack:
            assert total_next_obs.shape == (roll_steps, n_envs, *settings['InputDimRND'])

        # 2. normalize intrinsic reward
        total_rewards_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                          total_int_rewards])
        mean, std, count = np.mean(total_rewards_per_env), np.std(total_rewards_per_env), len(total_rewards_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        if rew_norm:
            total_int_rewards /= np.sqrt(reward_rms.var)
        # ------------------------------------

        # 3. compute target and advantages
        total_ext_target, ext_adv = compute_target_adv(total_rewards, total_dones, total_ext_values,
                                                       float(settings['Gamma']), roll_steps, n_envs)
        total_int_target, int_adv = compute_target_adv(total_int_rewards, np.zeros_like(total_int_rewards),
                                                       total_int_values, float(settings['Gamma']), roll_steps, n_envs)
        total_adv = int_adv * float(settings['IntCoef']) + ext_adv * float(settings['ExtCoef'])
        # ------------------------------------

        # 4. update observations normalization
        if obs_gray_stack:
            observation_rms.update(total_next_obs.reshape(-1, *settings['InputDimRND']))
        else:
            observation_rms.update(total_states_.reshape(-1, *settings['InputDim']))
        # ------------------------------------

        # 5. train agent
        if obs_norm:
            if obs_gray_stack:
                total_next_states = ((total_next_obs.reshape(-1, *settings['InputDimRND']) - observation_rms.mean) /
                                     np.sqrt(observation_rms.var)).clip(-5, 5)
            else:
                total_next_states = ((total_states_.reshape(-1, *settings['InputDim']) - observation_rms.mean) /
                                     np.sqrt(observation_rms.var)).clip(-5, 5)
        else:
            if obs_gray_stack:
                total_next_states = total_next_obs
            else:
                total_next_states = total_states_

        agent.learn(total_states, total_ext_target, total_int_target, total_actions,
                    total_adv, total_next_states, total_policies)
        # ------------------------------------

        # 6. wandb log
        if WANDB:
            wandb.log({
                'Scores/Int-Reward': np.sum(total_int_rewards) / n_envs,
                'Scores/Ext-Reward': np.sum(total_rewards) / n_envs,
                'Normalization Parameters/Reward-rms-mean': reward_rms.mean,
                'Normalization Parameters/Reward-rms-var': reward_rms.var,
                'Normalization Parameters/Observation-rms-mean': np.sum(observation_rms.mean),
                'Normalization Parameters/Observation-rms-var': np.sum(observation_rms.var)
            })
        # ------------------------------------

        # 7. save model
        if global_update % 35 == 0:
            print('\n... saving models ...')
            torch.save(agent.model.state_dict(), f'models/agent_{model_id}.pt')
            torch.save(agent.rnd.predictor.state_dict(), f'models/predictor_{model_id}.pt')
            torch.save(agent.rnd.target.state_dict(), f'models/target_{model_id}.pt')
            model_id += 1
            if model_id == 10:
                model_id = 0
