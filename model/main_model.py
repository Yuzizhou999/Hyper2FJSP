import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import nonzero_averaging
from model.attention_layer import MultiHeadMchAttnBlock, MultiHeadOpAttnBlock
from model.mo_attention_layer import MOMultiHeadMchAttnBlock, MOMultiHeadOpAttnBlock
from model.sub_layers import Actor, Critic, HyperActor


class DualAttentionNetwork(nn.Module):
    def __init__(self, config, fea_j_input_dim=None, fea_m_input_dim=None):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        :param fea_j_input_dim: dynamically adjusted input dimension for fea_j
        :param fea_m_input_dim: dynamically adjusted input dimension for fea_m
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = (
            fea_j_input_dim if fea_j_input_dim is not None else config.fea_j_input_dim
        )
        self.fea_m_input_dim = (
            fea_m_input_dim if fea_m_input_dim is not None else config.fea_m_input_dim
        )
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        mid_dim = self.output_dim_per_layer[:-1]
        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i]
                    * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i]
                    * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = (
                candidate.unsqueeze(-1).repeat(1, 1, fea_j.shape[-1]).type(torch.int64)
            )

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul, fea_j_jc).reshape(
                sz_b, M, M, -1
            )
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1,
        ).to(device)
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1,
        ).to(device)

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
    ):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :param preferences: preference vector with shape [sz_b, pref_dim]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
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
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
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

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v


class MO_DANIEL_ENC_WEIGHT_INPUT(DANIEL):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(MO_DANIEL_ENC_WEIGHT_INPUT, self).__init__(config)

        # Get preference dimension from objective functions
        pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

        # Dynamically adjust input dimensions for fea_j and fea_m
        fea_j_input_dim = config.fea_j_input_dim + pref_dim
        fea_m_input_dim = config.fea_m_input_dim + pref_dim

        # Initialize the feature extraction network with adjusted dimensions
        self.feature_exact = DualAttentionNetwork(
            config=config,
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
        ).to(config.device)

        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else pref_dim,
        ).to(config.device)

    def forward(  # type: ignore
        self,
        fea_j,
        op_mask,
        candidate,
        fea_m,
        mch_mask,
        comp_idx,
        dynamic_pair_mask,
        fea_pairs,
        preferences,
        fea_j_params=None,
        fea_m_params=None,
    ):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :param preferences: preference vector with shape [sz_b, pref_dim]
        :param fea_j_params: learnable job/operation feature parameters for active search [num_preferences, N_operations, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per operation
        :param fea_m_params: learnable machine feature parameters for active search [num_preferences, N_machines, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per machine
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1] or [sz_b, pref_dim]
        """
        fea_j = torch.cat(
            (fea_j, preferences.unsqueeze(1).repeat(1, fea_j.size(1), 1)), dim=-1
        )
        fea_m = torch.cat(
            (fea_m, preferences.unsqueeze(1).repeat(1, fea_m.size(1), 1)), dim=-1
        )
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
        )

        # Add learnable parameters if provided (for active search)
        if fea_j_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_j_params.shape[0]
            sz_b = fea_j.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_j.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_j_params: [num_prefs, N_operations, hidden_dim]
            # selected_fea_j_params: [sz_b, N_operations, hidden_dim]
            selected_fea_j_params = fea_j_params[preference_indices]

            # Add to per-operation features (direct addition, no broadcasting needed)
            # fea_j: [sz_b, N_operations, hidden_dim] + [sz_b, N_operations, hidden_dim]
            fea_j = fea_j + selected_fea_j_params

            # For global features, average the parameters across operations
            # fea_j_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_j_global = fea_j_global + selected_fea_j_params.mean(dim=1)

        if fea_m_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_m_params.shape[0]
            sz_b = fea_m.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_m.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_m_params: [num_prefs, N_machines, hidden_dim]
            # selected_fea_m_params: [sz_b, N_machines, hidden_dim]
            selected_fea_m_params = fea_m_params[preference_indices]

            # Add to per-machine features (direct addition, no broadcasting needed)
            # fea_m: [sz_b, N_machines, hidden_dim] + [sz_b, N_machines, hidden_dim]
            fea_m = fea_m + selected_fea_m_params

            # For global features, average the parameters across machines
            # fea_m_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_m_global = fea_m_global + selected_fea_m_params.mean(dim=1)

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
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
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
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

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v


class MODualAttentionNetworkOpAndMch(nn.Module):
    def __init__(
        self,
        config,
        fea_j_input_dim=None,
        fea_m_input_dim=None,
        return_pref_embeddings=False,
        use_gamma_beta=True,
    ):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        :param fea_j_input_dim: dynamically adjusted input dimension for fea_j
        :param fea_m_input_dim: dynamically adjusted input dimension for fea_m
        :param return_pref_embeddings: flag to return preference embeddings
        :param use_gamma_beta: whether to use gamma/beta transformation on input features
        """
        super(MODualAttentionNetworkOpAndMch, self).__init__()

        self.return_pref_embeddings = return_pref_embeddings
        self.use_gamma_beta = use_gamma_beta

        self.fea_j_input_dim = (
            fea_j_input_dim if fea_j_input_dim is not None else config.fea_j_input_dim
        )
        self.fea_m_input_dim = (
            fea_m_input_dim if fea_m_input_dim is not None else config.fea_m_input_dim
        )
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MOMultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                    use_gamma_beta=use_gamma_beta,
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MOMultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i]
                    * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i]
                    * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=(
                        nn.ELU()
                        if i < self.num_dan_layers - 1
                        else self.last_layer_activate
                    ),
                    dropout_prob=self.dropout_prob,
                    use_gamma_beta=use_gamma_beta,
                )
            )

    def forward(
        self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pref_j, pref_m
    ):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param pref_j: preference embeddings for operations with shape [sz_b, N, pref_dim]
        :param pref_m: preference embeddings for machines with shape [sz_b, M, pref_dim]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = (
                candidate.unsqueeze(-1).repeat(1, 1, fea_j.shape[-1]).type(torch.int64)
            )

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul, fea_j_jc).reshape(
                sz_b, M, M, -1
            )
            pref_m_comp_val = comp_val_layer.view(sz_b, M * M, -1).mean(dim=-2)
            fea_j, pref_j = self.op_attention_blocks[layer](fea_j, op_mask, pref_j)
            fea_m, pref_m = self.mch_attention_blocks[layer](
                fea_m, mch_mask, comp_val_layer, pref_m, pref_m_comp_val
            )

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        if self.return_pref_embeddings:
            return fea_j, fea_m, fea_j_global, fea_m_global, pref_j, pref_m
        else:
            return fea_j, fea_m, fea_j_global, fea_m_global


class MODANIELConditionalOpAndMch(nn.Module):
    def __init__(
        self, config, fea_j_input_dim=None, fea_m_input_dim=None, use_gamma_beta=True
    ):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        :param fea_j_input_dim: dynamically adjusted input dimension for fea_j
        :param fea_m_input_dim: dynamically adjusted input dimension for fea_m
        :param use_gamma_beta: whether to use gamma/beta transformation
        """
        super(MODANIELConditionalOpAndMch, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = MODualAttentionNetworkOpAndMch(
            config, use_gamma_beta=use_gamma_beta
        ).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1,
        ).to(device)
        self.single_value_critic = config.single_value_critic
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else pref_dim,
        ).to(config.device)

        # Get preference dimension from objective functions
        pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

        self.linear_transform_preferences_j = nn.Linear(
            pref_dim, config.fea_j_input_dim
        ).to(device)
        self.linear_transform_preferences_m = nn.Linear(
            pref_dim, config.fea_m_input_dim
        ).to(device)

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
        preferences,
        fea_j_params=None,
        fea_m_params=None,
    ):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :param preferences: preference vector with shape [sz_b, pref_dim]
        :param fea_j_params: learnable job/operation feature parameters for active search [num_preferences, N_operations, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per operation
        :param fea_m_params: learnable machine feature parameters for active search [num_preferences, N_machines, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per machine
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        pref_j = self.linear_transform_preferences_j(preferences)
        pref_m = self.linear_transform_preferences_m(preferences)
        pref_j = pref_j.unsqueeze(1).repeat(1, fea_j.size(1), 1)
        pref_m = pref_m.unsqueeze(1).repeat(1, fea_m.size(1), 1)
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pref_j, pref_m
        )

        # Add learnable parameters if provided (for active search)
        if fea_j_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_j_params.shape[0]
            sz_b = fea_j.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_j.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_j_params: [num_prefs, N_operations, hidden_dim]
            # selected_fea_j_params: [sz_b, N_operations, hidden_dim]
            selected_fea_j_params = fea_j_params[preference_indices]

            # Add to per-operation features (direct addition, no broadcasting needed)
            # fea_j: [sz_b, N_operations, hidden_dim] + [sz_b, N_operations, hidden_dim]
            fea_j = fea_j + selected_fea_j_params

            # For global features, average the parameters across operations
            # fea_j_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_j_global = fea_j_global + selected_fea_j_params.mean(dim=1)

        if fea_m_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_m_params.shape[0]
            sz_b = fea_m.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_m.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_m_params: [num_prefs, N_machines, hidden_dim]
            # selected_fea_m_params: [sz_b, N_machines, hidden_dim]
            selected_fea_m_params = fea_m_params[preference_indices]

            # Add to per-machine features (direct addition, no broadcasting needed)
            # fea_m: [sz_b, N_machines, hidden_dim] + [sz_b, N_machines, hidden_dim]
            fea_m = fea_m + selected_fea_m_params

            # For global features, average the parameters across machines
            # fea_m_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_m_global = fea_m_global + selected_fea_m_params.mean(dim=1)

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
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
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
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

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v


class MODANIELConditionalOpAndMchFea_Input(nn.Module):
    def __init__(
        self, config, fea_j_input_dim=None, fea_m_input_dim=None, use_gamma_beta=True
    ):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        :param fea_j_input_dim: dynamically adjusted input dimension for fea_j
        :param fea_m_input_dim: dynamically adjusted input dimension for fea_m
        :param use_gamma_beta: whether to use gamma/beta transformation
        """
        super(MODANIELConditionalOpAndMchFea_Input, self).__init__()
        device = torch.device(config.device)

        # Get preference dimension from objective functions
        pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        # Dynamically adjust input dimensions for fea_j and fea_m based on preference dimension
        fea_j_input_dim = config.fea_j_input_dim + pref_dim
        fea_m_input_dim = config.fea_m_input_dim + pref_dim

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = MODualAttentionNetworkOpAndMch(
            config,
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            use_gamma_beta=use_gamma_beta,
        ).to(device)
        #     config,
        # ).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1,
        ).to(device)
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else pref_dim,
        ).to(config.device)

        self.linear_transform_preferences_j = nn.Linear(pref_dim, fea_j_input_dim).to(
            device
        )
        self.linear_transform_preferences_m = nn.Linear(pref_dim, fea_m_input_dim).to(
            device
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
        preferences,
        fea_j_params=None,
        fea_m_params=None,
    ):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :param preferences: preference vector with shape [sz_b, pref_dim]
        :param fea_j_params: learnable job/operation feature parameters for active search [num_preferences, N_operations, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per operation
        :param fea_m_params: learnable machine feature parameters for active search [num_preferences, N_machines, hidden_dim]
                            Used to enable gradient-based optimization of preference-specific features per machine
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        pref_j = self.linear_transform_preferences_j(preferences)
        pref_m = self.linear_transform_preferences_m(preferences)
        pref_j = pref_j.unsqueeze(1).repeat(1, fea_j.size(1), 1)
        pref_m = pref_m.unsqueeze(1).repeat(1, fea_m.size(1), 1)
        fea_j = torch.cat(
            (fea_j, preferences.unsqueeze(1).repeat(1, fea_j.size(1), 1)), dim=-1
        )
        fea_m = torch.cat(
            (fea_m, preferences.unsqueeze(1).repeat(1, fea_m.size(1), 1)), dim=-1
        )
        #     fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pref_j, pref_m
        # )
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pref_j, pref_m
        )

        # Add learnable parameters if provided (for active search)
        if fea_j_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_j_params.shape[0]
            sz_b = fea_j.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_j.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_j_params: [num_prefs, N_operations, hidden_dim]
            # selected_fea_j_params: [sz_b, N_operations, hidden_dim]
            selected_fea_j_params = fea_j_params[preference_indices]

            # Add to per-operation features (direct addition, no broadcasting needed)
            # fea_j: [sz_b, N_operations, hidden_dim] + [sz_b, N_operations, hidden_dim]
            fea_j = fea_j + selected_fea_j_params

            # For global features, average the parameters across operations
            # fea_j_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_j_global = fea_j_global + selected_fea_j_params.mean(dim=1)

        if fea_m_params is not None:
            # Determine preference indices if not provided
            num_prefs = fea_m_params.shape[0]
            sz_b = fea_m.shape[0]
            preference_indices = torch.arange(sz_b, device=fea_m.device) % num_prefs

            # Select appropriate parameter tensor for each sample
            # fea_m_params: [num_prefs, N_machines, hidden_dim]
            # selected_fea_m_params: [sz_b, N_machines, hidden_dim]
            selected_fea_m_params = fea_m_params[preference_indices]

            # Add to per-machine features (direct addition, no broadcasting needed)
            # fea_m: [sz_b, N_machines, hidden_dim] + [sz_b, N_machines, hidden_dim]
            fea_m = fea_m + selected_fea_m_params

            # For global features, average the parameters across machines
            # fea_m_global: [sz_b, hidden_dim] + [sz_b, hidden_dim]
            fea_m_global = fea_m_global + selected_fea_m_params.mean(dim=1)

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
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
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
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

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v


# HYPER_DANIEL is maintained in its own file and re-exported here for
# backward compatibility.
from model.hyper_model import HYPER_DANIEL  # noqa: E402,F401
