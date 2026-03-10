"""
基于超网络的 DANIEL 多目标学习模型。

本模块包含 HYPER_DANIEL 模型，该模型将标准的双注意力网络（DualAttentionNetwork）
编码器与 HyperActor 结合，实现偏好条件化的策略生成。
这是论文中的 HYPER 架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention_layer import MultiHeadMchAttnBlock, MultiHeadOpAttnBlock
from model.hyper_network import AttentionPooling, HyperActor
from model.sub_layers import Critic
from model.main_model import DualAttentionNetwork


class HYPER_DANIEL(nn.Module):
    def __init__(self, config):
        """
            基于超网络的 DANIEL 多目标学习框架实现
        :param config: 参数配置包
        """
        super(HYPER_DANIEL, self).__init__()
        device = torch.device(config.device)

        # 成对特征输入维度（固定值）
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        # 使用与 DANIEL 相同的特征提取网络
        self.feature_exact = DualAttentionNetwork(config).to(device)

        # 根据目标函数数量确定偏好维度
        pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )
        self.use_instance_features = getattr(
            config, "hyper_use_instance_features", False
        )

        # 使用可学习的注意力权重替代 nonzero_averaging，为全局特征提供更具表达力的聚合方式
        self.attn_pool_j = AttentionPooling(self.embedding_output_dim).to(device)
        self.attn_pool_m = AttentionPooling(self.embedding_output_dim).to(device)

        # 将问题实例的全局特征注入超网络，使生成的 Actor 参数不仅取决于偏好向量，还取决于当前问题实例的特征
        instance_dim = (
            2 * self.embedding_output_dim if self.use_instance_features else 0
        )

        # 增加 instance_dim 支持实例感知超网络
        self.actor = HyperActor(
            num_layers=config.num_mlp_layers_actor,
            input_dim=4 * self.embedding_output_dim + self.pair_input_dim,
            hidden_dim=config.hidden_dim_actor,
            output_dim=1,
            pref_dim=pref_dim,
            hyper_hidden_dim=config.hyper_hidden_dim,
            embd_dim=config.hyper_embd_dim,
            instance_dim=instance_dim,
        ).to(device)

        # 存储当前偏好；仅在 preference-only 模式下预生成 Actor 参数
        self.current_preferences = None

        # Critic 配置
        self.single_value_critic = config.single_value_critic
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else pref_dim,
        ).to(device)

    def assign_preferences(self, preferences):
        """
        存储偏好向量。
        在 preference-only 模式下可直接预生成 Actor 参数；
        在实例感知模式下，需要等 forward() 获得实例特征后再生成。

        :param preferences: 偏好向量 [batch_size, pref_dim]
                           批次中的每个样本可以有不同的偏好
        """
        self.current_preferences = preferences
        if not self.use_instance_features:
            self.actor.assign(preferences)

    def forward(
        self,
        fea_j,
        op_mask,
        candidate,
        fea_m,
        mch_mask,
        comp_idx,
        dynamic_pair_mask,
        fea_pairs,
        preferences=None,
        preference_indices=None,
        fea_j_params=None,
        fea_m_params=None,
    ):
        """
        :param candidate: 候选操作的索引，形状 [sz_b, J]
        :param fea_j: 输入操作特征向量，形状 [sz_b, N, 8]
        :param op_mask: 用于掩码不存在的前驱/后继操作（形状 [sz_b, N, 3]）
        :param fea_m: 输入机器特征向量，形状 [sz_b, M, 6]
        :param mch_mask: 用于掩码注意力系数（形状 [sz_b, M, M]）
        :param comp_idx: 形状 [sz_b, M, M, J] 的张量，用于计算 T_E
                    comp_idx[i, k, q, j]（任意 i）表示机器 $M_k$ 和 $M_q$
                    是否在竞争 candidate[i,j]
        :param dynamic_pair_mask: 形状 [sz_b, J, M] 的张量，用于掩码不兼容的操作-机器对
        :param fea_pairs: 成对特征，形状 [sz_b, J, M, 8]
        :param preferences: 偏好向量 [batch_size, pref_dim]
                           PPO 更新时传入；rollout 时为 None（使用 assign_preferences 存储的偏好）
        :param preference_indices: 将每个样本映射到已存储偏好的索引 [batch_size]
                                    rollout 时用于从 current_preferences 中选取对应偏好
        :param fea_j_params: 用于主动搜索的可学习作业特征参数 [num_preferences, hidden_dim]
                            用于启用基于梯度的偏好特定特征优化
        :param fea_m_params: 用于主动搜索的可学习机器特征参数 [num_preferences, hidden_dim]
                            用于启用基于梯度的偏好特定特征优化
        :return:
            pi: 调度策略，形状 [sz_b, J*M]
            v: 状态价值，形状 [sz_b, 1] 或 [sz_b, pref_dim]
        """
        # 特征提取（与 DANIEL 相同）
        fea_j, fea_m, _, _ = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )

        # 如果提供了可学习参数，则添加到特征中（用于主动搜索）
        if fea_j_params is not None:
            # 如果未提供偏好索引，则自动确定
            if preference_indices is None:
                num_prefs = fea_j_params.shape[0]
                sz_b = fea_j.shape[0]
                preference_indices = torch.arange(sz_b, device=fea_j.device) % num_prefs

            selected_fea_j_params = fea_j_params[preference_indices]
            fea_j = fea_j + selected_fea_j_params

        if fea_m_params is not None:
            if preference_indices is None:
                num_prefs = fea_m_params.shape[0]
                sz_b = fea_m.shape[0]
                preference_indices = torch.arange(sz_b, device=fea_m.device) % num_prefs

            selected_fea_m_params = fea_m_params[preference_indices]
            fea_m = fea_m + selected_fea_m_params

        # 使用注意力池化计算全局特征，让模型可学习地决定各节点对全局表示的贡献
        fea_j_global = self.attn_pool_j(fea_j)  # [sz_b, d]
        fea_m_global = self.attn_pool_m(fea_m)  # [sz_b, d]

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # 收集决策网络的输入
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = (
            Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        candidate_feature = torch.cat(
            (
                Fea_j_JC_serialized,
                Fea_m_serialized,
                Fea_Gj_input,
                Fea_Gm_input,
                fea_pairs,
            ),
            dim=-1,
        )

        # 实例感知超网络,将注意力池化得到的全局特征作为实例信息注入超网络，使 Actor 参数同时取决于偏好向量和当前问题实例
        instance_global = torch.cat([fea_j_global, fea_m_global], dim=-1)  # [sz_b, 2*d]

        # 确定当前批次使用的偏好向量
        if preferences is not None:
            # PPO 更新阶段：偏好由调用者显式传入
            batch_prefs = preferences
        elif self.current_preferences is not None:
            # Rollout 阶段：使用 assign_preferences() 存储的偏好
            if preference_indices is not None:
                batch_prefs = self.current_preferences[preference_indices]
            else:
                batch_prefs = self.current_preferences
        else:
            raise RuntimeError(
                "必须在前向传播前调用 assign_preferences() 或传入 preferences 参数！"
            )

        actor_preference_indices = None
        if self.use_instance_features:
            self.actor.assign(batch_prefs, instance_features=instance_global)
        elif preferences is not None:
            self.actor.assign(batch_prefs)
        elif preference_indices is not None:
            actor_preference_indices = preference_indices

        candidate_scores = self.actor(
            candidate_feature, preference_indices=actor_preference_indices
        )
        candidate_scores = candidate_scores.squeeze(-1)

        # 掩码不兼容的操作-机器对
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        # Critic 前向传播
        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
