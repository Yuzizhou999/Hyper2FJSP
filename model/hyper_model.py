"""
基于超网络的多目标 DANIEL 变体。

该模块保留了原始的 HyperActor，并通过可选的编码器端调节功能进行了扩展，
使得偏好信号既可以影响决策头（Actor），也可以影响编码器的部分结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hyper_network import (
    AttentionPooling,
    EventHyperAdapter,
    HyperActor,
    HyperEncoderConditioner,
)
from model.sub_layers import Critic


class HYPER_DANIEL(nn.Module):
    """
    基于超网络的多目标调度模型 (HYPER_DANIEL)。
    结合了对偶注意力网络进行特征提取，并通过超网络动态生成网络参数
    以适应不同的多目标优化偏好。
    """
    def __init__(self, config):
        """
        初始化 HYPER_DANIEL 模型。
        
        :param config: 模型的配置项对象，包含各类超参数结构。
        """
        super(HYPER_DANIEL, self).__init__()
        device = torch.device(config.device)

        # 节点对的输入特征维度
        self.pair_input_dim = 8
        # 特征提取后的节点嵌入维度
        self.embedding_output_dim = config.layer_fea_output_dim[-1]
        # 偏好向量的维度（目标数）
        self.pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

        # 各种高级特性的开启标志位
        self.use_instance_features = getattr(
            config, "hyper_use_instance_features", False
        )
        self.condition_encoder = getattr(config, "hyper_condition_encoder", False)
        self.hyper_encoder_use_film = getattr(
            config, "hyper_encoder_use_film", True
        )
        self.hyper_encoder_use_attention = getattr(
            config, "hyper_encoder_use_attention", True
        )
        self.use_event_hypernetwork = bool(
            getattr(config, "hyper_event_adapter_enabled", False)
        )
        
        # 事件上下文的维度
        self.event_context_dim = 10
        self.supports_event_context = self.use_event_hypernetwork

        from model.main_model import DualAttentionNetwork, MODualAttentionNetworkOpAndMch

        # 根据配置选用支持条件输入的编码器或基础双层注意力网络
        if self.condition_encoder and self.hyper_encoder_use_attention:
            self.feature_exact = MODualAttentionNetworkOpAndMch(
                config, use_gamma_beta=config.use_gamma_beta
            ).to(device)
        else:
            self.feature_exact = DualAttentionNetwork(config).to(device)

        # 工件与机器的注意力池化层，用于提取全局特征
        self.attn_pool_j = AttentionPooling(self.embedding_output_dim).to(device)
        self.attn_pool_m = AttentionPooling(self.embedding_output_dim).to(device)

        # 编码器条件调节器，生成 FiLM 调节参数和注意力偏移嵌入
        self.encoder_conditioner = None
        if self.condition_encoder:
            self.encoder_conditioner = HyperEncoderConditioner(
                pref_dim=self.pref_dim,
                fea_j_input_dim=config.fea_j_input_dim,
                fea_m_input_dim=config.fea_m_input_dim,
                hyper_hidden_dim=config.hyper_hidden_dim,
            ).to(device)

        # 确定提供给超网络的实例特征维度
        instance_dim = (
            2 * self.embedding_output_dim if self.use_instance_features else 0
        )
        
        # HyperActor: 由超网络动态给定权重的 Actor模块
        self.actor = HyperActor(
            num_layers=config.num_mlp_layers_actor,
            input_dim=4 * self.embedding_output_dim + self.pair_input_dim,
            hidden_dim=config.hidden_dim_actor,
            output_dim=1,
            pref_dim=self.pref_dim,
            hyper_hidden_dim=config.hyper_hidden_dim,
            embd_dim=config.hyper_embd_dim,
            instance_dim=instance_dim,
        ).to(device)
        
        # 事件级环境适配器，根据当前事件生成Actor权重更新步长(delta)
        self.event_adapter = None
        if self.use_event_hypernetwork:
            self.event_adapter = EventHyperAdapter(
                event_dim=self.event_context_dim,
                input_dim=4 * self.embedding_output_dim + self.pair_input_dim,
                hidden_dim=config.hidden_dim_actor,
                output_dim=1,
                adapter_hidden_dim=config.hyper_event_hidden_dim,
                delta_scale=config.hyper_event_delta_scale,
            ).to(device)

        # 缓存当前的偏好权重
        self.current_preferences = None
        
        # 带有独立网络权重的 Critic，用于估值 (可输出单一值也可多目标)
        self.single_value_critic = config.single_value_critic
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else self.pref_dim,
        ).to(device)

    def assign_preferences(self, preferences):
        """
        预先计算并向所有条件模块分配偏好参数。
        
        :param preferences: 需要被分配的偏好向量组 [B, pref_dim]
        """
        self.current_preferences = preferences.detach()
        with torch.no_grad():
            if self.encoder_conditioner is not None:
                self.encoder_conditioner.assign(self.current_preferences)
            # 如果不使用实例特征则可以提前将基础的Actor权重推断出来
            # 否则必须在执行 forward 的阶段等待实例输入特征一同注入
            if not self.use_instance_features:
                self.actor.assign(self.current_preferences)

    def _resolve_preferences(self, preferences, preference_indices):
        """
        处理偏好的取用逻辑。如果在前向传播中未显式传入偏好，则根据当前
        预缓存的偏好或者由索引取出指定偏好。
        """
        if preferences is not None:
            return preferences
        if self.current_preferences is None:
            raise RuntimeError(
                "请先在 forward() 前调用 assign_preferences() 分配偏好，"
                "或者在前向时直接传入 preferences。"
            )
        if preference_indices is not None:
            # 如果使用重用的缓存，通过索引用到的偏好做快速映射
            return self.current_preferences[preference_indices]
        return self.current_preferences

    def _encode_features(
        self,
        fea_j,
        op_mask,
        candidate,
        fea_m,
        mch_mask,
        comp_idx,
        preferences,
        preference_indices,
    ):
        """
        对工件(Job)特征与机器(Machine)特征进行环境编码提取，同时可应用 FiLM 及偏好嵌入。
        
        :param fea_j: 工件节点特征
        :param op_mask: 工序屏蔽掩码
        :param candidate: 候选工序状态
        :param fea_m: 机器节点特征
        :param mch_mask: 机器屏蔽掩码
        :param comp_idx: 将候选工序与机器匹配的复合索引
        :param preferences: 偏好参数矩阵
        :param preference_indices: 用于直接选择偏好的缓存索引
        :return: 经过注意力网等信息交互后的 H_j 与 H_m 隐特征
        """
        # 如果未启用编码器条件调节器，采用原始模型的编码结构
        if self.encoder_conditioner is None:
            return self.feature_exact(
                fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
            )

        # 存在编码器调节参数，准备相应变长参数对以便前向应用
        conditioner_kwargs = {}
        if preferences is not None:
            conditioner_kwargs["preferences"] = preferences
        else:
            conditioner_kwargs["preference_indices"] = preference_indices

        # 根据偏好信息生成经过 FiLM 调节层的嵌入映射，和专用的偏好信号
        conditioned_fea_j, conditioned_fea_m, pref_j, pref_m = self.encoder_conditioner(
            fea_j, fea_m, **conditioner_kwargs
        )

        # 使用基于仿射的条件信号（FiLM）控制是否真的应用调节在原始特征上
        encoder_fea_j = conditioned_fea_j if self.hyper_encoder_use_film else fea_j
        encoder_fea_m = conditioned_fea_m if self.hyper_encoder_use_film else fea_m

        if self.hyper_encoder_use_attention:
            return self.feature_exact(
                encoder_fea_j,
                op_mask,
                candidate,
                encoder_fea_m,
                mch_mask,
                comp_idx,
                pref_j,
                pref_m,
            )

        return self.feature_exact(
            encoder_fea_j, op_mask, candidate, encoder_fea_m, mch_mask, comp_idx
        )

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
        event_context=None,
        event_mch_context=None,
    ):
        """
        模型的前向传播逻辑。
        
        :param fea_j: 工件节点特征矩阵
        :param op_mask: 工序有效掩码
        :param candidate: 当前可选计算的工序
        :param fea_m: 机器节点特征矩阵
        :param mch_mask: 机器有效掩码
        :param comp_idx: 将工序与其可行机器映射在一起的索引网络
        :param dynamic_pair_mask: 动态的（工序, 机器）二元对连边掩码
        :param fea_pairs: （工序, 机器）级别的配对特征
        :param preferences: 执行时即时传入的偏好向量（可选）
        :param preference_indices: 用于查询内部所生成偏好的索引（可选）
        :param fea_j_params: 直接调整提取出来的工件特征的额外残差（可选）
        :param fea_m_params: 直接调整提取出来的机器特征的额外残差（可选）
        :param event_context: 动态调度事件环境的全局事件上下文特征（可选）
        :param event_mch_context: 动态调度中机器专属的事件上下文（暂未单独使用）
        :return: (
            pi: 代表动作选择概率分布的张量，
            v: Critic网络对给定状态求出的估值
        )
        """
        # 1. 决定和验证要使用的偏好信息
        batch_prefs = self._resolve_preferences(preferences, preference_indices)

        # 2. 调用具体的编码结构提取出包含当前调度状态和偏好的信息隐层表征
        fea_j, fea_m, _, _ = self._encode_features(
            fea_j,
            op_mask,
            candidate,
            fea_m,
            mch_mask,
            comp_idx,
            preferences=batch_prefs if preferences is not None else None,
            preference_indices=None if preferences is not None else preference_indices,
        )

        # 追加外部注入的残差修正(如有)
        if fea_j_params is not None:
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

        # 3. 产生用于评估和执行的全局层面环境信息（Attention Pooling）
        fea_j_global = self.attn_pool_j(fea_j)
        fea_m_global = self.attn_pool_m(fea_m)

        # 准备对每个“侯选工序-机器”配对做动作概率预估
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # 4. 根据candidate抽出对应的有效工件节点隐态
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        fea_j_jc = torch.gather(fea_j, 1, candidate_idx)

        # 拉平化，变成 [Batch, Machine*Job, D] 以与配对信息拼接
        fea_j_jc_serialized = (
            fea_j_jc.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(
            sz_b, M * J, d
        )

        # 将全局特征拓展为每一对可用形式
        fea_gj_input = fea_j_global.unsqueeze(1).expand_as(fea_j_jc_serialized)
        fea_gm_input = fea_m_global.unsqueeze(1).expand_as(fea_j_jc_serialized)

        # 拼接配对相关的独属特征，最终得到可以交给Actor打分的配对级特征集合
        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        candidate_feature = torch.cat(
            (
                fea_j_jc_serialized,
                fea_m_serialized,
                fea_gj_input,
                fea_gm_input,
                fea_pairs,
            ),
            dim=-1,
        )

        # 5. 组合实例级别的全局特征用于超网络注入（如有开启）
        instance_global = torch.cat([fea_j_global, fea_m_global], dim=-1)

        actor_preference_indices = None
        # 如果开启动态实例特征感知，需要在当此 forward 直接生成 Actor 的基于此特定环境的独属参数
        if self.use_instance_features:
            self.actor.assign(batch_prefs, instance_features=instance_global)
        # 如果是即时计算的偏好，即使不基于局部特征也需要进行超网络一次前推分配
        elif preferences is not None:
            self.actor.assign(batch_prefs)
        # 如果既不使用动态实例也不即时分配，直接拿分配时的缓存使用索引用法取件
        elif preference_indices is not None:
            actor_preference_indices = preference_indices

        # 6. 计算事件相关的残差偏置项 (Delta)
        event_deltas = {}
        if self.event_adapter is not None:
            if event_context is None:
                # 默认情况下赋予一个空特征
                event_context = torch.zeros(
                    (candidate_feature.shape[0], self.event_context_dim),
                    device=candidate_feature.device,
                    dtype=candidate_feature.dtype,
                )
            if event_context.dim() == 1:
                event_context = event_context.unsqueeze(0)
            if event_context.shape[-1] != self.event_context_dim:
                raise RuntimeError(
                    "事件上下文(event_context) 的维度与设置中的 event_context_dim 不匹配"
                )
            # 拿到用于叠加在其内部参数上的Delta变动量
            event_deltas = self.event_adapter(event_context)

        # 7. 交给 HyperActor 给每个可能的“工序-机器”执行评分
        candidate_scores = self.actor(
            candidate_feature,
            preference_indices=actor_preference_indices,
            **event_deltas,
        ).squeeze(-1)
        
        # 使用动态配对掩码过滤掉不合法的调度配对
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        # 归一化为选择动作的概率分布
        pi = F.softmax(candidate_scores, dim=1)

        # 8. 全局维度再拼接出一维传入 Critic 求价值预测
        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        
        return pi, v
