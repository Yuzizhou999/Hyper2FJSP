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
            用于在 PPO 训练中收集并存储轨迹数据的记忆库
        :param gamma: 折扣因子 (discount factor)
        :param gae_lambda: PPO 算法中的广义优势估计 (GAE) 参数
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # DANIEL 模型的输入变量
        self.fea_j_seq = []  # 工序特征序列 [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # 工序掩码序列 [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # 机器特征序列 [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # 机器掩码序列 [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # 动态配对掩码序列 [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # 竞争兼容性索引序列 [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # 候选工序序列 [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # 配对特征序列 [N, tensor[sz_b, J]]
        self.event_context_seq = []  # 动态事件上下文特征序列 [N, tensor[sz_b, event_dim]]

        # 其他强化学习训练所需变量
        self.action_seq = []  # 动作索引，形状为 [N, tensor[sz_b]]
        self.reward_seq = []  # 奖励值，形状为 [N, tensor[sz_b]]
        self.val_seq = []  # 状态价值即 Critic 网络的评估值，形状为 [N, tensor[sz_b]]
        self.done_seq = []  # 终止标志位 done flag，形状为 [N, tensor[sz_b]]
        self.log_probs = []  # 动作的对数概率 log(p_{\theta_old}(a_t|s_t))，形状为 [N, tensor[sz_b]]

    def clear_memory(self):
        """清空所有存储的轨迹和回滚数据。"""
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        """只清空状态相关的特征张量，保留动作和回报分布等原有数据。"""
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]
        del self.event_context_seq[:]

    def push(self, state):
        """
            将当前的马尔可夫决策过程（MDP）状态推入记忆库中保存
        :param state: 被收集保存的 MDP state 对象
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
        self.event_context_seq.append(state.event_context_tensor)

    def transpose_data(self):
        """
        翻转已收集变量的第一维与第二维。由[时间步、环境批次]转为[环境批次、时间步并展平]
        
        :return: 经过扁平化处理且对齐过的张量元组，供 PPO 算法采用 batch-first 更新
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
        t_event_context_seq = (
            torch.stack(self.event_context_seq, dim=0).transpose(0, 1).flatten(0, 1)
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
            t_event_context_seq,
            t_action_seq,
            t_reward_seq,
            t_val_seq,
            t_done_seq,
            t_logprobs_seq,
        )

    def get_gae_advantages(self, objective_weights=None):
        """
            计算被广泛使用的广义优势估计值（Generalized Advantage Estimates, GAE）
        :param objective_weights: 偏好权重参数，用以标量化多目标场景；单目标设为None即可
        :return: 算得的优势估计序列 (advantage sequences) 和更新前真实的目标状态值序列 (state value sequence)
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
            PPO 算法的核心实现模块
        :param config: 全局参数配置实例
        :param multi_objective: 标志当前算法是否应用于多目标优化场景
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

        # 根据配置的网络架构以及是否多目标，实例化相应的策略网络
        if multi_objective:
            if (
                config.model_architecture
                == ModelArchitecture.MO_DANIEL_ENC_WEIGHT_INPUT
            ):
                self.policy = MO_DANIEL_ENC_WEIGHT_INPUT(config)
            # 采用特征加权或者条件机制变体的多目标网络
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
            # 引入超网络（Hypernetwork）参数动态生成的架构
            elif config.model_architecture == ModelArchitecture.HYPER_DANIEL:
                self.policy = HYPER_DANIEL(config)
            else:
                # 默认使用包含权重输入的多目标网络
                print(
                    f"Warning: Unknown model architecture {config.model_architecture} for multi-objective. Using MO_DANIEL_ENC_WEIGHT_INPUT."
                )
                self.policy = MO_DANIEL_ENC_WEIGHT_INPUT(config)
        else:
            if (
                config.model_architecture == ModelArchitecture.DANIEL
                or config.model_architecture.value.startswith("mo_")
            ):
                # 哪怕配置里写着多目标，只要实例化成单目标参数，就退化为基座 DANIEL 网络
                self.policy = DANIEL(config)
            else:
                # 单目标场景的备选防故障默认网络
                print(
                    f"Warning: Unknown model architecture {config.model_architecture} for single-objective. Using DANIEL."
                )
                self.policy = DANIEL(config)

        # 构建 PPO 更新时的“旧网络”副本用于比对信任区域 (Trust Region)  
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def _clip_gradients(self):
        """如果明确定义了最大梯度阈值，则实施梯度截断操作，防止梯度爆炸"""
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
        基于当前收集的片段缓冲执行多周期的 PPO 裁剪更新
        :param memory: 用于训练该批次的经验池数据数据对象
        :param objective_weights: 多目标优化每个观测维度的特定权重偏好 (如果为单目标则为 None)
        :param single_value_critic: 判别多目标场景下 Critic 评价标量结果是被合成为单个标量还是针对每一维保留的旗帜变量
        :return: 每周期平均的主网总损失 和 评论家价值网络 Critic 更新损失
        """
        obj_weight_input = objective_weights is not None
        t_data = memory.transpose_data()

        # 如果有传入偏好权重并且针对所有目标独立评分
        if obj_weight_input and not single_value_critic:
            t_advantage_seq, v_target_seq = memory.get_gae_advantages(
                objective_weights=objective_weights
            )
        else:
            t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        expanded_preferences = None
        if obj_weight_input:
            # 扩展偏好权重，使得它们与批次内推平后的环境个数相对齐
            expanded_preferences = objective_weights.repeat_interleave(
                t_data[0].shape[0] // objective_weights.shape[0], dim=0
            )

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        loss_epochs = 0
        v_loss_epochs = 0

        # 进行 K_epochs 轮迭代以复用这些被收集好的固定训练数据片段
        for _ in range(self.k_epochs):
            # 将收集好的数据全集随机打乱索引以提升收敛稳定性
            permutation = torch.randperm(full_batch_size, device=t_data[0].device)

            # 由于显存受限，把全量数据切片为多个 minibatch（微小批次）分别训练更新
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    # 分割出最后一批数据的尾部收敛范围
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
                
                # 如果开启并存在动态事件上下文张量，将它放入主模型的调用传参中
                if (
                    hasattr(self.policy, "supports_event_context")
                    and self.policy.supports_event_context
                    and t_data[8].shape[-1] > 0
                ):
                    policy_inputs["event_context"] = t_data[8][batch_indices]
                if expanded_preferences is not None:
                    policy_inputs["preferences"] = expanded_preferences[batch_indices]

                # 对此 minibatch 在现更新中再推演评价一次获取分布及新的 Critic value 预测值
                pis, vals = self.policy(**policy_inputs)

                action_batch = t_data[9][batch_indices]
                # 计算针对此前已经下发采取的旧动作在新概率分布下的话语权和熵奖励
                logprobs, ent_loss = eval_actions(pis, action_batch)
                
                # ratios = 经过指数变换的新对数概率减去收集时的旧对数概率 = p_{new}(a_t|s_t) / p_{old}(a_t|s_t)
                ratios = torch.exp(logprobs - t_data[13][batch_indices].detach())

                advantages = t_advantage_seq[batch_indices]
                # PPO 第一个非截断替代目标 (Surrogate Objective)
                surr1 = ratios * advantages
                # PPO 第二个防策略坍塌并具备信赖域裁剪的替代目标 (Clipped Surrogate Objective)
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
                # 策略梯度损失 (负均值因为优化器试图最小化损失)    
                p_loss = -torch.min(surr1, surr2)
                ent_loss = -ent_loss.clone() # 负向熵是为了增加梯度探索性
                
                # PPO 三部分损失全合并计算：[价值损失] + [策略削减损失] + [高熵随机激励（探索项）]
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
                
        # 更新策略网络的影子副本 (policy_old)，这里采用缓慢软更新 Soft Update 的方式渐进迁移
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
