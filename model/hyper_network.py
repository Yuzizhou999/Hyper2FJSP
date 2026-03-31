"""
基于超网络的 Actor 模块。

本模块包含 HyperActor 类，该类利用超网络从偏好向量动态生成
Actor 网络的参数。这是 HYPER 模型架构的核心组件。

同时包含 AttentionPooling 模块，用可学习的注意力权重替代简单均值池化，
为全局特征提供更具表达力的聚合方式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    基于注意力的池化模块，替代 nonzero_averaging。

    使用可学习的小型 MLP 计算每个节点的重要性权重，
    自动屏蔽零向量（已删除节点），然后通过加权求和生成全局特征。
    """

    def __init__(self, d, hidden_dim=None):
        """
        :param d: 输入特征维度
        :param hidden_dim: 注意力 MLP 的隐藏层维度（默认等于 d）
        """
        super(AttentionPooling, self).__init__()
        hidden_dim = hidden_dim or d
        self.attn = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        :param x: 特征向量 [sz_b, N, d]（零向量表示已删除节点）
        :return: 加权聚合后的全局特征 [sz_b, d]
        """
        # 计算注意力分数
        scores = self.attn(x).squeeze(-1)  # [sz_b, N]

        # 屏蔽零向量（填充节点）
        mask = x.abs().sum(dim=-1) == 0  # [sz_b, N]
        scores = scores.masked_fill(mask, float("-inf"))

        # softmax 注意力权重
        weights = F.softmax(scores, dim=-1)  # [sz_b, N]

        # 处理全部被屏蔽的情况（避免 NaN）
        weights = weights.masked_fill(mask, 0.0)

        # 加权求和
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [sz_b, d]


class HyperEncoderConditioner(nn.Module):
    """
    基于偏好生成轻量级的编码器条件信号。

    该调节器仅为原始编码器输入生成 FiLM（特征线性调制）参数，
    以及生成供条件注意力编码器使用的偏好嵌入。
    """

    def __init__(
        self,
        pref_dim,
        fea_j_input_dim,
        fea_m_input_dim,
        hyper_hidden_dim=256,
        film_scale_limit=0.5,
        film_shift_limit=0.5,
    ):
        super(HyperEncoderConditioner, self).__init__()
        self.fea_j_input_dim = fea_j_input_dim
        self.fea_m_input_dim = fea_m_input_dim
        self.film_scale_limit = film_scale_limit
        self.film_shift_limit = film_shift_limit
        self.hyper_activation = nn.ReLU()

        self.hyper_fc1 = nn.Linear(pref_dim, hyper_hidden_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_dim, hyper_hidden_dim, bias=True)

        self.j_gamma_head = nn.Linear(hyper_hidden_dim, fea_j_input_dim)
        self.j_beta_head = nn.Linear(hyper_hidden_dim, fea_j_input_dim)
        self.m_gamma_head = nn.Linear(hyper_hidden_dim, fea_m_input_dim)
        self.m_beta_head = nn.Linear(hyper_hidden_dim, fea_m_input_dim)

        self.pref_j_head = nn.Linear(hyper_hidden_dim, fea_j_input_dim)
        self.pref_m_head = nn.Linear(hyper_hidden_dim, fea_m_input_dim)

        # 用于缓存推断一次后的参数
        self.cached_j_gamma = None
        self.cached_j_beta = None
        self.cached_m_gamma = None
        self.cached_m_beta = None
        self.cached_pref_j = None
        self.cached_pref_m = None

    def assign(self, pref):
        """预先计算一次并缓存从偏好到FiLM和编码器额外参数的生成结果"""
        if pref.dim() == 1:
            pref = pref.unsqueeze(0)

        h = self.hyper_activation(self.hyper_fc1(pref))
        h = self.hyper_activation(self.hyper_fc2(h))

        self.cached_j_gamma = 1.0 + self.film_scale_limit * torch.tanh(
            self.j_gamma_head(h)
        )
        self.cached_j_beta = self.film_shift_limit * torch.tanh(self.j_beta_head(h))
        self.cached_m_gamma = 1.0 + self.film_scale_limit * torch.tanh(
            self.m_gamma_head(h)
        )
        self.cached_m_beta = self.film_shift_limit * torch.tanh(self.m_beta_head(h))
        self.cached_pref_j = self.pref_j_head(h)
        self.cached_pref_m = self.pref_m_head(h)
        return pref.shape[0]

    def _select(self, tensor, preference_indices):
        """如果前向传播传入了索引用法，按索引取缓存，否则直接拿原缓存"""
        return tensor if preference_indices is None else tensor[preference_indices]

    def forward(self, fea_j, fea_m, preferences=None, preference_indices=None):
        """
        前向时将缓存在调节器里的仿射参数加在输入的 Job / Machine 初始特征上。
        """
        if preferences is not None:
            self.assign(preferences)

        if self.cached_j_gamma is None:
            raise RuntimeError(
                "编码器条件生成模块还未被初始化。"
                "请先对其执行 assign(pref) 操作，或者直接传 preferences 进来！"
            )

        j_gamma = self._select(self.cached_j_gamma, preference_indices)
        j_beta = self._select(self.cached_j_beta, preference_indices)
        m_gamma = self._select(self.cached_m_gamma, preference_indices)
        m_beta = self._select(self.cached_m_beta, preference_indices)
        pref_j = self._select(self.cached_pref_j, preference_indices)
        pref_m = self._select(self.cached_pref_m, preference_indices)

        if fea_j.shape[0] != j_gamma.shape[0] or fea_m.shape[0] != m_gamma.shape[0]:
            raise RuntimeError(
                "编码器条件模块的 Batch Size 与特征输入的 Batch Size 不匹配。"
                "在重用缓存分配的偏好时可以使用 preference_indices 进行正确映射。"
            )

        fea_j = fea_j * j_gamma.unsqueeze(1) + j_beta.unsqueeze(1)
        fea_m = fea_m * m_gamma.unsqueeze(1) + m_beta.unsqueeze(1)
        pref_j = pref_j.unsqueeze(1).expand(-1, fea_j.size(1), -1)
        pref_m = pref_m.unsqueeze(1).expand(-1, fea_m.size(1), -1)

        return fea_j, fea_m, pref_j, pref_m


class EventHyperAdapter(nn.Module):
    """
    为 HyperActor 的参数生成基于事件条件的残差（补偿信号）。
    在动态调度环境下，根据事件上下文（Event Context）动态调整已经由偏好生成的Actor网络参数。
    """

    def __init__(
        self,
        event_dim,
        input_dim,
        hidden_dim,
        output_dim,
        adapter_hidden_dim=128,
        delta_scale=0.1,
    ):
        """
        初始化事件超网络适配器。
        
        :param event_dim: 事件上下文特征向量的维度
        :param input_dim: Actor网络的输入维度
        :param hidden_dim: Actor网络的隐藏层维度
        :param output_dim: Actor网络的输出维度
        :param adapter_hidden_dim: 适配器内部MLP的隐藏层维度
        :param delta_scale: 放缩因子，用于控制残差更新的幅度
        """
        super(EventHyperAdapter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.delta_scale = delta_scale

        total_delta_dim = (
            hidden_dim * input_dim
            + hidden_dim
            + hidden_dim * hidden_dim
            + hidden_dim
            + output_dim * hidden_dim
            + output_dim
        )

        self.fc1 = nn.Linear(event_dim, adapter_hidden_dim)
        self.fc2 = nn.Linear(adapter_hidden_dim, adapter_hidden_dim)
        self.fc_out = nn.Linear(adapter_hidden_dim, total_delta_dim)
        self.gate = nn.Linear(adapter_hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, event_context):
        """
        前向传播生成因事件而异的网络残差。
        
        :param event_context: 描述当前动态调度事件上下文的特征向量
        :return: 包含 3 层网络参数偏移量(delta)的字典
        """
        if event_context.dim() == 1:
            event_context = event_context.unsqueeze(0)

        # 1. 对上下文进行非线性特征化
        h = self.activation(self.fc1(event_context))
        h = self.activation(self.fc2(h))
        
        # 2. 计算门控值控制偏移影响
        gate = torch.sigmoid(self.gate(h))
        
        # 3. 计算最终的放缩残差张量
        delta = self.delta_scale * torch.tanh(self.fc_out(h)) * gate

        batch_size = event_context.shape[0]
        cursor = 0

        # 分配第 1 层参数
        lin1_size = self.hidden_dim * self.input_dim
        bias1_size = self.hidden_dim
        # 分配第 2 层参数
        lin2_size = self.hidden_dim * self.hidden_dim
        bias2_size = self.hidden_dim
        # 分配第 3 层参数
        lin3_size = self.output_dim * self.hidden_dim
        bias3_size = self.output_dim

        # 4. 根据相应结构的尺寸拆分并重塑 (resaphe) 为适配的权重矩阵格式
        lin1_delta = delta[:, cursor : cursor + lin1_size].reshape(
            batch_size, self.hidden_dim, self.input_dim
        )
        cursor += lin1_size
        bias1_delta = delta[:, cursor : cursor + bias1_size].reshape(
            batch_size, self.hidden_dim
        )
        cursor += bias1_size
        lin2_delta = delta[:, cursor : cursor + lin2_size].reshape(
            batch_size, self.hidden_dim, self.hidden_dim
        )
        cursor += lin2_size
        bias2_delta = delta[:, cursor : cursor + bias2_size].reshape(
            batch_size, self.hidden_dim
        )
        cursor += bias2_size
        lin3_delta = delta[:, cursor : cursor + lin3_size].reshape(
            batch_size, self.output_dim, self.hidden_dim
        )
        cursor += lin3_size
        bias3_delta = delta[:, cursor : cursor + bias3_size].reshape(
            batch_size, self.output_dim
        )

        return {
            "dec_lin1_delta": lin1_delta,
            "dec_bias1_delta": bias1_delta,
            "dec_lin2_delta": lin2_delta,
            "dec_bias2_delta": bias2_delta,
            "dec_lin3_delta": lin3_delta,
            "dec_bias3_delta": bias3_delta,
        }


class HyperActor(nn.Module):
    """
    基于超网络的 Actor，通过偏好向量生成网络参数。
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        pref_dim,
        hyper_hidden_dim=256,
        embd_dim=2,
        instance_dim=0,
    ):
        """
        :param num_layers: Actor 网络的层数（应为 3）
        :param input_dim: 输入特征的维度
        :param hidden_dim: 所有层的隐藏单元维度
        :param output_dim: 输出维度（Actor 通常为 1）
        :param pref_dim: 偏好向量的维度（例如 2 个目标时为 2）
        :param hyper_hidden_dim: 超网络的隐藏层维度（默认: 256）
        :param embd_dim: 参数生成的嵌入维度（默认: 2）
        :param instance_dim: 实例全局特征的维度（默认: 0，即不使用实例感知）
                            当 > 0 时，超网络通过残差注入接收问题实例信息
        """
        super(HyperActor, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pref_dim = pref_dim
        self.embd_dim = embd_dim
        self.instance_dim = instance_dim
        self.activative = torch.tanh
        self.hyper_activation = nn.ReLU()

        if num_layers != 3:
            raise ValueError("HyperActor 目前仅支持 3 层！")

        # 超网络：从偏好向量生成嵌入
        # 架构: pref_dim -> 256 -> 256 -> (embd_dim * 6)
        self.hyper_output_dim = (
            self.embd_dim * 6
        )  # 6 = 3 层 × 2 个参数（权重、偏置）
        self.hyper_fc1 = nn.Linear(pref_dim, hyper_hidden_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_dim, hyper_hidden_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_dim, self.hyper_output_dim, bias=True)

        # 实例特征注入：将问题实例的全局特征投影到超网络隐藏空间
        # 通过残差加法注入到第一隐藏层输出，不改变原有偏好处理路径
        if instance_dim > 0:
            self.instance_proj = nn.Linear(instance_dim, hyper_hidden_dim, bias=True)

        # 参数生成器：将嵌入转换为实际网络参数
        # 第 1 层: input_dim -> hidden_dim
        self.hyper_lin1 = nn.Linear(self.embd_dim, input_dim * hidden_dim)
        self.hyper_bias1 = nn.Linear(self.embd_dim, hidden_dim)

        # 第 2 层: hidden_dim -> hidden_dim
        self.hyper_lin2 = nn.Linear(self.embd_dim, hidden_dim * hidden_dim)
        self.hyper_bias2 = nn.Linear(self.embd_dim, hidden_dim)

        # 第 3 层: hidden_dim -> output_dim
        self.hyper_lin3 = nn.Linear(self.embd_dim, hidden_dim * output_dim)
        self.hyper_bias3 = nn.Linear(self.embd_dim, output_dim)

        # 存储生成的参数（由 assign 方法设置）
        self.dec_lin1_para = None
        self.dec_bias1_para = None
        self.dec_lin2_para = None
        self.dec_bias2_para = None
        self.dec_lin3_para = None
        self.dec_bias3_para = None

    def assign(self, pref, instance_features=None):
        """
        从偏好向量（及可选的实例特征）生成网络参数。
        支持单个偏好和批量偏好。

        :param pref: 偏好向量
                    - 单个: [pref_dim]
                    - 批量: [batch_size, pref_dim]
        :param instance_features: 实例全局特征（可选）
                    - 单个: [instance_dim]
                    - 批量: [batch_size, instance_dim]
                    当提供时，通过残差加法注入超网络第一隐藏层
        :return: batch_size（单个偏好返回 1，批量偏好返回 batch_size）
        """
        # 确保 pref 具有批次维度
        if pref.dim() == 1:
            pref = pref.unsqueeze(0)  # [pref_dim] -> [1, pref_dim]

        batch_size = pref.shape[0]

        # 通过超网络为批次中所有偏好生成嵌入
        # pref: [batch_size, pref_dim]
        h = self.hyper_fc1(pref)  # [batch_size, hyper_hidden_dim]

        # 实例特征残差注入
        if instance_features is not None and self.instance_dim > 0:
            if instance_features.dim() == 1:
                instance_features = instance_features.unsqueeze(0)
            h = h + self.instance_proj(instance_features)

        h = self.hyper_activation(h)
        h = self.hyper_activation(self.hyper_fc2(h))
        mid_embd = self.hyper_fc3(h)
        # mid_embd: [batch_size, embd_dim * 6]

        # 沿最后一维分割为 6 个大小为 embd_dim 的块
        mid_embd = mid_embd.chunk(6, dim=-1)  # 6 个 [batch_size, embd_dim] 的张量

        # 为每一层生成参数
        # 第 1 层: input_dim -> hidden_dim
        self.dec_lin1_para = self.hyper_lin1(mid_embd[0]).reshape(
            batch_size, self.hidden_dim, self.input_dim
        )  # [batch_size, hidden_dim, input_dim]
        self.dec_bias1_para = self.hyper_bias1(mid_embd[1]).reshape(
            batch_size, self.hidden_dim
        )  # [batch_size, hidden_dim]

        # 第 2 层: hidden_dim -> hidden_dim
        self.dec_lin2_para = self.hyper_lin2(mid_embd[2]).reshape(
            batch_size, self.hidden_dim, self.hidden_dim
        )  # [batch_size, hidden_dim, hidden_dim]
        self.dec_bias2_para = self.hyper_bias2(mid_embd[3]).reshape(
            batch_size, self.hidden_dim
        )  # [batch_size, hidden_dim]

        # 第 3 层: hidden_dim -> output_dim
        self.dec_lin3_para = self.hyper_lin3(mid_embd[4]).reshape(
            batch_size, self.output_dim, self.hidden_dim
        )  # [batch_size, output_dim, hidden_dim]
        self.dec_bias3_para = self.hyper_bias3(mid_embd[5]).reshape(
            batch_size, self.output_dim
        )  # [batch_size, output_dim]

        return batch_size

    def forward(
        self,
        x,
        preferences=None,
        preference_indices=None,
        dec_lin1_delta=None,
        dec_bias1_delta=None,
        dec_lin2_delta=None,
        dec_bias2_delta=None,
        dec_lin3_delta=None,
        dec_bias3_delta=None,
    ):
        """
        使用生成的参数进行前向传播。
        批次中的每个样本使用其对应的生成参数。

        :param x: 输入特征 [batch_size, seq_len, input_dim]
        :param preferences: 偏好向量（可选，用于即时重新生成参数）
                           [batch_size, pref_dim] 或 None
        :param preference_indices: 将每个样本映射到参数集中对应偏好的索引
                                    [batch_size]，每个值的范围为 [0, num_unique_prefs)
                                    提供此参数时使用索引参数选择以提高内存效率
        :return: 输出分数 [batch_size, seq_len, output_dim]
        """
        # 如果提供了偏好向量，则重新生成参数
        if preferences is not None:
            self.assign(preferences)

        if self.dec_lin1_para is None:
            raise RuntimeError(
                "必须在前向传播前调用 assign(pref) 或提供 preferences 参数！"
            )

        # 使用生成的参数进行 3 层网络计算
        # 对于批量参数，需要将每个样本的参数应用于其对应的输入

        # 如果提供了 preference_indices，为每个样本选择正确的参数
        if preference_indices is not None:
            # 使用索引选择参数
            # dec_lin1_para: [num_unique_prefs, hidden_dim, input_dim]
            # preference_indices: [batch_size]
            # 结果: [batch_size, hidden_dim, input_dim]
            lin1_para = self.dec_lin1_para[preference_indices]
            bias1_para = self.dec_bias1_para[preference_indices]
            lin2_para = self.dec_lin2_para[preference_indices]
            bias2_para = self.dec_bias2_para[preference_indices]
            lin3_para = self.dec_lin3_para[preference_indices]
            bias3_para = self.dec_bias3_para[preference_indices]
        else:
            # 直接使用参数（标准批量方式）
            lin1_para = self.dec_lin1_para
            bias1_para = self.dec_bias1_para
            lin2_para = self.dec_lin2_para
            bias2_para = self.dec_bias2_para
            lin3_para = self.dec_lin3_para
            bias3_para = self.dec_bias3_para

        # 第 1 层: tanh 激活
        # 使用批量矩阵乘法 bmm 进行线性运算
        # x: [batch_size, seq_len, input_dim]
        # lin1_para: [batch_size, hidden_dim, input_dim]
        # 需要: [batch_size, seq_len, input_dim] @ [batch_size, input_dim, hidden_dim]

        if dec_lin1_delta is not None:
            lin1_para = lin1_para + dec_lin1_delta
        if dec_bias1_delta is not None:
            bias1_para = bias1_para + dec_bias1_delta
        if dec_lin2_delta is not None:
            lin2_para = lin2_para + dec_lin2_delta
        if dec_bias2_delta is not None:
            bias2_para = bias2_para + dec_bias2_delta
        if dec_lin3_delta is not None:
            lin3_para = lin3_para + dec_lin3_delta
        if dec_bias3_delta is not None:
            bias3_para = bias3_para + dec_bias3_delta

        h = torch.bmm(x, lin1_para.transpose(1, 2))  # [batch_size, seq_len, hidden_dim]
        h = h + bias1_para.unsqueeze(1)  # 加偏置: [batch_size, 1, hidden_dim]
        h = torch.tanh(h)

        # 第 2 层: tanh 激活
        # h: [batch_size, seq_len, hidden_dim]
        # lin2_para: [batch_size, hidden_dim, hidden_dim]
        h = torch.bmm(h, lin2_para.transpose(1, 2))  # [batch_size, seq_len, hidden_dim]
        h = h + bias2_para.unsqueeze(1)  # 加偏置: [batch_size, 1, hidden_dim]
        h = torch.tanh(h)

        # 第 3 层: 无激活函数（线性输出）
        # h: [batch_size, seq_len, hidden_dim]
        # lin3_para: [batch_size, output_dim, hidden_dim]
        h = torch.bmm(h, lin3_para.transpose(1, 2))  # [batch_size, seq_len, output_dim]
        h = h + bias3_para.unsqueeze(1)  # 加偏置: [batch_size, 1, output_dim]

        return h
