from typing import Tuple

import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from models import PPOModel, RNDModel
from utils import global_grad_norm, check_grad


class RNDAgent(object):
    """
    PPO Agent with Random Network Distillation.

    ...

    Attributes
    ----------
    settings : dict
        list of hyperparameters
    model : PPOModel
        Proximal Policy Optimization model
    rnd : RNDModel
        Random Network Distillation model
    optimizer: torch.optim
        ADAM optimizer
    device: torch.device
        GPU if available else CPU

    Methods
    -------
    get_action(state):
        Returns action, values, policy from the current state.
    compute_intrinsic_reward(state_):
        Returns intrinsic reward
    learn(s_batch, te_batch, ti_batch, a_batch, adv_batch, ns_batch, p_batch)
        Train the agent
    """

    def __init__(self, settings: dict):
        """
        Constructs all the necessary attributes for the RNDAgent object.

        :param settings: list of hyperparameters
        """
        self.settings = settings

        self.model = PPOModel(self.settings['Envs'], self.settings['InputDim'], self.settings['Actions'])
        self.rnd = RNDModel(self.settings['Envs'], self.settings['InputDimRND'], int(self.settings['EncodingSize']))

        self.optimizer = opt.Adam(list(self.model.parameters()) + list(self.rnd.parameters()),
                                  lr=float(self.settings['LearningRate']))

        self.device = torch.device('cuda:0' if bool(self.settings['UseCuda']) and torch.cuda.is_available
                                   else 'cpu')
        print(f'DEVICE: {self.device}')

        self.model.to(self.device)
        self.rnd.to(self.device)

    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Returns action, values, policy from current state(s)

        :param state: current state(s)
        :return: action, value_ext, value_int, policy
        """
        state = torch.Tensor(state).to(self.device)
        state = state.float()

        output_data = self.model(state)
        probs = F.softmax(output_data['policy'], dim=-1)
        dist = Categorical(probs)
        action = dist.sample().cpu().detach().numpy()
        value_ext = output_data['value_ext'].cpu().detach().numpy().squeeze()
        value_int = output_data['value_int'].cpu().detach().numpy().squeeze()
        policy = output_data['policy'].detach()

        return action, value_ext, value_int, policy

    def compute_intrinsic_reward(self, next_state: np.ndarray) -> np.ndarray:
        """
        compute intrinsic reward for a given next_state

        :param next_state: next state
        :return: intrinsic reward
        """
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_state_features = self.rnd(next_state)
        # target features
        target_features = next_state_features['target_features']
        # predictor features
        predictor_features = next_state_features['predictor_features']
        # intrinsic reward
        intrinsic_reward = (target_features - predictor_features).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def learn(self, state_batch: np.ndarray, target_ext_batch: np.ndarray,
              target_int_batch: np.ndarray, action_batch: np.ndarray,
              advantage_batch: np.ndarray, next_state_batch: np.ndarray,
              policy_batch):
        """

        :param state_batch: batch of states
        :param target_ext_batch: batch of external targets
        :param target_int_batch: batch of internal targets
        :param action_batch: batch of actions
        :param advantage_batch: batch of total advantages (external + internal)
        :param next_state_batch: batch of next states
        :param policy_batch: batch of policy

        :return: None
        """
        state_batch = torch.FloatTensor(state_batch).reshape(-1, *self.settings['InputDim']).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).reshape(-1).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).reshape(-1).to(self.device)
        action_batch = torch.FloatTensor(action_batch).reshape(-1).to(self.device)
        advantage_batch = torch.FloatTensor(advantage_batch).reshape(-1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).reshape(-1, *self.settings['InputDimRND']).to(self.device)

        mse_loss = nn.MSELoss(reduction='none')

        # get old log_prob
        with torch.no_grad():
            policy_old = policy_batch.contiguous().reshape(-1, self.settings['Actions']).to(self.device)
            probs_old = Categorical(F.softmax(policy_old, dim=-1))
            log_prob_old = probs_old.log_prob(action_batch)

        sample_range = np.arange(state_batch.shape[0])

        policy_loss, critic_loss, entropy_loss, rnd_loss, overall_loss = 0, 0, 0, 0, 0
        for i in range(int(self.settings['Epochs'])):

            np.random.shuffle(sample_range)

            for j in range(int(state_batch.shape[0] / self.settings['BatchSize'])):
                sample_idx = sample_range[self.settings['BatchSize'] * j:self.settings['BatchSize'] * (j + 1)]

                # random network distillation loss
                next_states_features = self.rnd(next_state_batch[sample_idx])
                rnd_loss = mse_loss(next_states_features['predictor_features'],
                                    next_states_features['target_features'].detach()).mean(-1)
                # --------------------------------

                # predictor proportional update
                mask = torch.rand(len(rnd_loss)).to(self.device)
                mask = (mask < float(self.settings['UpdateRate'])).type(torch.FloatTensor).to(self.device)
                rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # --------------------------------

                # ppo update
                model_output = self.model(state_batch[sample_idx])
                policies = model_output['policy']
                values_ext = model_output['value_ext']
                values_int = model_output['value_int']
                probs = Categorical(F.softmax(policies, dim=-1))
                log_prob = probs.log_prob(action_batch[sample_idx])

                prob_ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                weighted_probs = advantage_batch[sample_idx] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio,
                                                     1.0 - float(self.settings['PolicyClip']),
                                                     1.0 + float(self.settings['PolicyClip'])) \
                                         * advantage_batch[sample_idx]
                # --------------------------------

                # agent loss
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_ext_loss = F.mse_loss(values_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(values_int.sum(1), target_int_batch[sample_idx])
                critic_loss = critic_ext_loss + critic_int_loss

                # entropy loss
                entropy_loss = probs.entropy().mean()
                # --------------------------------

                # overall loss
                overall_loss = policy_loss + 0.5 * critic_loss - float(self.settings['Entropy']) * entropy_loss \
                               + rnd_loss

                # reset gradients
                self.optimizer.zero_grad()

                # backpropagation
                overall_loss.backward()
                global_grad_norm(list(self.model.parameters()) + list(self.rnd.predictor.parameters()))

                # optimization step
                self.optimizer.step()

                # check null gradients
                check_grad(self.model.named_parameters())
                check_grad(self.rnd.predictor.named_parameters())

        if self.settings['WandB']:
            wandb.log({
               'Losses/Policy-Loss': policy_loss,
               'Losses/Critic-Loss': critic_loss,
               'Losses/Entropy-Loss': entropy_loss,
               'Losses/RND-Loss': rnd_loss,
               'Losses/Overall-Loss': overall_loss,
               'Gradients/Grad-Norm-Agent': global_grad_norm(list(self.model.parameters())),
               'Gradients/Grad-Norm-Predictor': global_grad_norm(list(self.rnd.predictor.parameters())),
            })
