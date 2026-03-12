from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common_utils import eval_actions
from enums import ModelArchitecture
from model.main_model import (
    DANIEL,
    HYPER_DANIEL,
    MO_DANIEL_ENC_WEIGHT_INPUT,
    MODANIELConditionalOpAndMch,
    MODANIELConditionalOpAndMchFea_Input,
)
from params import configs


class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]

    def clear_memory(self):
        """Remove all stored trajectories and rollouts."""
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        """Remove only the stored state-related tensors, keeping actions/returns intact."""
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory
        :param state: the MDP state
        :return:
        """
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        """
        Transpose the first and second dimension of collected variables.

        :return: tuple of flattened tensors aligned for batch-first PPO updates
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = (
            torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = (
            torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_dynamicMask_seq = (
            torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_Compete_m_seq = (
            torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_candidate_seq = (
            torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_pairMessage_seq = (
            torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        )
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = (
            torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)
        )

        return (
            t_Fea_j_seq,
            t_op_mask_seq,
            t_Fea_m_seq,
            t_mch_mask_seq,
            t_dynamicMask_seq,
            t_Compete_m_seq,
            t_candidate_seq,
            t_pairMessage_seq,
            t_action_seq,
            t_reward_seq,
            t_val_seq,
            t_done_seq,
            t_logprobs_seq,
        )

    def get_gae_advantages(self, objective_weights=None):
        """
            Compute the generalized advantage estimates
        :param objective_weights: preference weights for objectives
        :return: advantage sequences, state value sequence
        """

        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        if objective_weights is not None:
            len_trajectory, len_envs, num_obj = reward_arr.shape
            advantage = torch.zeros(len_envs, device=values.device)
            advantage_val = torch.zeros((len_envs, num_obj), device=values.device)
            advantage_seq = []
            val_target_seq = []
            for i in reversed(range(len_trajectory)):
                if i == len_trajectory - 1:
                    # Use weighted sum
                    delta_t = (reward_arr[i] * objective_weights).sum(axis=1) - (
                        values[i] * objective_weights
                    ).sum(axis=1)
                    delta_t_val = reward_arr[i] - values[i]
                else:
                    # Use weighted sum
                    delta_t = (
                        (reward_arr[i] * objective_weights).sum(axis=1)
                        + self.gamma * (values[i + 1] * objective_weights).sum(axis=1)
                        - (values[i] * objective_weights).sum(axis=1)
                    )
                    delta_t_val = reward_arr[i] + self.gamma * values[i + 1] - values[i]
                advantage = delta_t + self.gamma * self.gae_lambda * advantage
                advantage_val = (
                    delta_t_val + self.gamma * self.gae_lambda * advantage_val
                )
                advantage_seq.insert(0, advantage)
                val_target_seq.insert(0, advantage_val)
            t_advantage_seq = (
                torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)
            )
            t_val_target_seq = (
                torch.stack(val_target_seq, dim=0).transpose(0, 1).to(torch.float32)
            )
            v_target_seq = (t_val_target_seq + self.t_old_val_seq).flatten(0, 1)

            # normalization
            t_advantage_seq = (
                t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)
            ) / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

            return t_advantage_seq.flatten(0, 1), v_target_seq

        else:
            len_trajectory, len_envs = reward_arr.shape

            advantage = torch.zeros(len_envs, device=values.device)
            advantage_seq = []
            for i in reversed(range(len_trajectory)):
                if i == len_trajectory - 1:
                    delta_t = reward_arr[i] - values[i]
                else:
                    delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
                advantage = delta_t + self.gamma * self.gae_lambda * advantage
                advantage_seq.insert(0, advantage)

            # [sz_b, N,]
            t_advantage_seq = (
                torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)
            )

            # [sz_b, N]
            v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

            # normalization
            t_advantage_seq = (
                t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)
            ) / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

            return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    def __init__(self, config, multi_objective: bool = False):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
        self.lr = config.lr
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.eps_clip = config.eps_clip
        self.k_epochs = config.k_epochs
        self.tau = config.tau

        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.minibatch_size = config.minibatch_size
        self.max_grad_norm = config.max_grad_norm

        # Choose model based on architecture and multi-objective flag
        if multi_objective:
            if (
                config.model_architecture
                == ModelArchitecture.MO_DANIEL_ENC_WEIGHT_INPUT
            ):
                self.policy = MO_DANIEL_ENC_WEIGHT_INPUT(config)
            # Conditional models with attention mechanism
            elif (
                config.model_architecture
                == ModelArchitecture.MO_DANIEL_CONDITIONAL_OP_AND_MCH
            ):
                self.policy = MODANIELConditionalOpAndMch(
                    config, use_gamma_beta=config.use_gamma_beta
                )
            elif (
                config.model_architecture
                == ModelArchitecture.MO_DANIEL_CONDITIONAL_OP_AND_MCH_FEA_INPUT
            ):
                self.policy = MODANIELConditionalOpAndMchFea_Input(
                    config, use_gamma_beta=config.use_gamma_beta
                )
            # Hypernetwork-based model
            elif config.model_architecture == ModelArchitecture.HYPER_DANIEL:
                self.policy = HYPER_DANIEL(config)
            else:
                # Default to encoder-based weight input for multi-objective
                print(
                    f"Warning: Unknown model architecture {config.model_architecture} for multi-objective. Using MO_DANIEL_ENC_WEIGHT_INPUT."
                )
                self.policy = MO_DANIEL_ENC_WEIGHT_INPUT(config)
        else:
            if (
                config.model_architecture == ModelArchitecture.DANIEL
                or config.model_architecture.value.startswith("mo_")
            ):
                # Use DANIEL for single-objective, even if a MO model was accidentally specified
                self.policy = DANIEL(config)
            else:
                # Default to DANIEL for any other model architecture in single-objective setting
                print(
                    f"Warning: Unknown model architecture {config.model_architecture} for single-objective. Using DANIEL."
                )
                self.policy = DANIEL(config)

        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def _clip_gradients(self):
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )

    def update(
        self,
        memory: Memory,
        objective_weights=None,
        single_value_critic=False,
    ):
        """
        :param memory: data used for PPO training
        :param objective_weights: optional preference weights per objective (None for single-objective)
        :param single_value_critic: whether to collapse multi-objective values into a single scalar critic
        :return: total_loss and critic_loss
        """
        obj_weight_input = objective_weights is not None
        t_data = memory.transpose_data()

        if obj_weight_input and not single_value_critic:
            t_advantage_seq, v_target_seq = memory.get_gae_advantages(
                objective_weights=objective_weights
            )
        else:
            t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        expanded_preferences = None
        if obj_weight_input:
            expanded_preferences = objective_weights.repeat_interleave(
                t_data[0].shape[0] // objective_weights.shape[0], dim=0
            )

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        loss_epochs = 0
        v_loss_epochs = 0

        for _ in range(self.k_epochs):
            permutation = torch.randperm(full_batch_size, device=t_data[0].device)

            # Split into multiple batches of updates due to memory limitations
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    # the last batch
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size

                batch_indices = permutation[start_idx:end_idx]
                policy_inputs = {
                    "fea_j": t_data[0][batch_indices],
                    "op_mask": t_data[1][batch_indices],
                    "candidate": t_data[6][batch_indices],
                    "fea_m": t_data[2][batch_indices],
                    "mch_mask": t_data[3][batch_indices],
                    "comp_idx": t_data[5][batch_indices],
                    "dynamic_pair_mask": t_data[4][batch_indices],
                    "fea_pairs": t_data[7][batch_indices],
                }
                if expanded_preferences is not None:
                    policy_inputs["preferences"] = expanded_preferences[batch_indices]

                pis, vals = self.policy(**policy_inputs)

                action_batch = t_data[8][batch_indices]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][batch_indices].detach())

                advantages = t_advantage_seq[batch_indices]
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                if obj_weight_input and not single_value_critic:
                    v_loss = self.V_loss_2(vals, v_target_seq[batch_indices])
                else:
                    v_loss = self.V_loss_2(
                        vals.squeeze(1), v_target_seq[batch_indices]
                    )
                p_loss = -torch.min(surr1, surr2)
                ent_loss = -ent_loss.clone()
                loss = (
                    self.vloss_coef * v_loss
                    + self.ploss_coef * p_loss
                    + self.entloss_coef * ent_loss
                )

                self.optimizer.zero_grad()
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                loss.mean().backward()
                self._clip_gradients()
                self.optimizer.step()
        # soft update
        for policy_old_params, policy_params in zip(
            self.policy_old.parameters(), self.policy.parameters()
        ):
            policy_old_params.data.copy_(
                self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data
            )

        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs


def PPO_initialize(multi_objective: bool):
    ppo = PPO(config=configs, multi_objective=multi_objective)
    return ppo
