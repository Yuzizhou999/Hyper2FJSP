import torch
import torch.nn as nn
import torch.nn.functional as F


class MOSingleOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, use_gamma_beta=True):
        """
            The implementation of Operation Message Attention Block
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()
        :param use_gamma_beta: whether to use gamma/beta transformation on input features
        """
        super(MOSingleOpAttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.use_gamma_beta = use_gamma_beta

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if use_gamma_beta:
            self.W_gamma = nn.Parameter(torch.empty(size=(input_dim, input_dim)))
            self.W_beta = nn.Parameter(torch.empty(size=(input_dim, input_dim)))
            self.W_gamma_beta = nn.Linear(input_dim, 2 * input_dim, bias=False)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, op_mask, pref):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        with shape [sz_b, N, 3]
        :param pref: preference vector with shape [sz_b, 2]
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        # Compute h' based on the preference vector
        if self.use_gamma_beta:
            gamma, beta = torch.chunk(self.W_gamma_beta(pref), 2, dim=-1)
            h_prime = h * gamma + beta
        else:
            h_prime = h

        Wh_prime = torch.matmul(h_prime, self.W)
        sz_b, N, _ = Wh_prime.size()

        Whpref = torch.matmul(pref, self.W)

        Wh_prime_concat = torch.stack(
            [Wh_prime.roll(1, dims=1), Wh_prime, Wh_prime.roll(-1, dims=1), Whpref],
            dim=-2,
        )
        Whpref_concat = torch.stack(
            [Wh_prime.roll(1, dims=1), Whpref, Wh_prime.roll(-1, dims=1), Wh_prime],
            dim=-2,
        )

        Wh1 = torch.matmul(Wh_prime, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh_prime, self.a[self.out_features :, :])

        Whpref1 = torch.matmul(Whpref, self.a[: self.out_features, :])
        Whpref2 = torch.matmul(Whpref, self.a[self.out_features :, :])

        Wh2_concat = torch.stack(
            [Wh2.roll(1, dims=1), Wh2, Wh2.roll(-1, dims=1), Whpref2], dim=-1
        )
        Whpref2_concat = torch.stack(
            [Wh2.roll(1, dims=1), Whpref2, Wh2.roll(-1, dims=1), Wh2], dim=-1
        )
        # broadcast add: [sz_b, N, 1, 1] + [sz_b, N, 1, 4]
        e = Wh1.unsqueeze(-1) + Wh2_concat
        e = self.leaky_relu(e)

        e_pref = Whpref1.unsqueeze(-1) + Whpref2_concat
        e_pref = self.leaky_relu(e_pref)

        zero_vec = -9e15 * torch.ones_like(e)
        op_mask = torch.cat(
            (op_mask, torch.zeros(sz_b, N, 1).to(op_mask.device)), dim=-1
        )
        attention = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        attention_pref = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e_pref)
        attention_pref = F.softmax(attention_pref, dim=-1)
        attention_pref = self.dropout(attention_pref)

        h_new = torch.matmul(attention, Wh_prime_concat).squeeze(-2)

        pref_new = torch.matmul(attention_pref, Whpref_concat).squeeze(-2)

        return h_new, pref_new


class MOMultiHeadOpAttnBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_prob,
        num_heads,
        activation,
        concat=True,
        use_gamma_beta=True,
    ):
        """
            The implementation of Operation Message Attention Block with multi-head attention
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        :param use_gamma_beta: whether to use gamma/beta transformation on input features
        """
        super(MOMultiHeadOpAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [
            MOSingleOpAttnBlock(input_dim, output_dim, dropout_prob, use_gamma_beta)
            for _ in range(num_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, h, op_mask, pref):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param pref: preference vector with shape [sz_b, pref_dim]
        :return: output feature vectors with shape
                [sz_b, N, num_heads * output_dim] (if concat == true)
                or [sz_b, N, output_dim]
        """
        h = self.dropout(h)

        # shape: [ [sz_b, N, output_dim], ... [sz_b, N, output_dim]]
        attention_outputs = [att(h, op_mask, pref) for att in self.attentions]

        h_heads = list(map(lambda x: x[0], attention_outputs))
        pref_heads = list(map(lambda x: x[1], attention_outputs))
        if self.concat:
            h = torch.cat(h_heads, dim=-1)
            pref = torch.cat(pref_heads, dim=-1)
        else:
            # h.shape : [sz_b, N, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            # h.shape : [sz_b, N, output_dim]
            h = h.mean(dim=-1)
            pref = torch.stack(pref_heads, dim=-1)
            pref = pref.mean(dim=-1)

        return (
            h if self.activation is None else self.activation(h),
            pref if self.activation is None else self.activation(pref),
        )


class MOSingleMchAttnBlock(nn.Module):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        output_dim,
        dropout_prob,
        use_gamma_beta=True,
    ):
        """
            The implementation of Machine Message Attention Block
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()
        :param use_gamma_beta: whether to use gamma/beta transformation on input features
        """
        super(MOSingleMchAttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.use_gamma_beta = use_gamma_beta

        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if use_gamma_beta:
            self.W_gamma_beta = nn.Linear(
                node_input_dim, 2 * node_input_dim, bias=False
            )

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, mch_mask, comp_val, pref, pref_comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :param pref: preference vector with shape [sz_b, pref_dim]
        :param pref_comp_val: a tensor with shape [sz_b, M, M, output_dim] used for additional preference values
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        # Apply gamma/beta transformation if enabled
        if self.use_gamma_beta:
            gamma, beta = torch.chunk(self.W_gamma_beta(pref), 2, dim=-1)
            h_prime = h * gamma + beta
        else:
            h_prime = h

        # Add the preference embedding as an extra node so machines can attend to preference signals directly
        h_prime = torch.concat((h_prime, pref[:, [0], :]), dim=1)

        Wh_prime = torch.matmul(h_prime, self.W)
        # Expand comp_val from [20, 5, 5, 11] to [20, 6, 6, 11]
        expanded_comp_val = torch.zeros(
            comp_val.shape[0],
            comp_val.shape[1] + 1,
            comp_val.shape[2] + 1,
            comp_val.shape[3],
            device=comp_val.device,
            dtype=comp_val.dtype,
        )

        # Copy the original comp_val into the top-left part
        expanded_comp_val[:, :-1, :-1, :] = comp_val

        # Add the preference tensor to the new row and column
        # For the new row (relations from preference node to all machines)
        expanded_comp_val[:, -1, :-1, :] = pref_comp_val.unsqueeze(1).expand(
            -1, comp_val.shape[2], -1
        )

        # For the new column (relations from all machines to preference node)
        expanded_comp_val[:, :-1, -1, :] = pref_comp_val.unsqueeze(1).expand(
            -1, comp_val.shape[1], -1
        )

        # Fill the bottom-right corner (preference node's self-relation)
        expanded_comp_val[:, -1, -1, :] = pref_comp_val
        comp_val = expanded_comp_val

        W_edge = torch.matmul(comp_val, self.W_edge)

        # compute attention matrix
        e = self.get_attention_coef(Wh_prime, W_edge)

        zero_vec = -9e15 * torch.ones_like(e)
        mch_mask = F.pad(mch_mask, (0, 1, 0, 1), "constant", 1)
        attention = torch.where(mch_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh_prime)
        h_pref = h_prime[:, -1, :].unsqueeze(1).expand(-1, pref.shape[1], -1)
        h_prime = h_prime[:, :-1, :]
        return h_prime, h_pref

    def get_attention_coef(self, Wh, W_edge):
        """
            compute attention coefficients using node and edge features
        :param Wh: transformed node features
        :param W_edge: transformed edge features
        :return:
        """

        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])  # [sz_b, M, 1]
        Wh2 = torch.matmul(
            Wh, self.a[self.out_features : 2 * self.out_features, :]
        )  # [sz_b, M, 1]
        edge_feas = torch.matmul(
            W_edge, self.a[2 * self.out_features :, :]
        )  # [sz_b, M, M, 1]

        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2) + edge_feas.squeeze(-1)

        return self.leaky_relu(e)


class MOMultiHeadMchAttnBlock(nn.Module):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        output_dim,
        dropout_prob,
        num_heads,
        activation,
        concat=True,
        use_gamma_beta=True,
    ):
        """
            The implementation of Machine Message Attention Block with multi-head attention
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        :param use_gamma_beta: whether to use gamma/beta transformation on input features
        """
        super(MOMultiHeadMchAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads

        self.attentions = [
            MOSingleMchAttnBlock(
                node_input_dim, edge_input_dim, output_dim, dropout_prob, use_gamma_beta
            )
            for _ in range(num_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, h, mch_mask, comp_val, pref_m, pref_m_comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] (any i) corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :param pref_m: preference vector for machines with shape [sz_b, M, pref_dim]
        :param pref_m_comp_val: a tensor with shape [sz_b, M, M, output_dim] used for additional preference values
        :return: output feature vectors with shape
                [sz_b, M, num_heads * output_dim] (if concat == true)
                or [sz_b, M, output_dim]
        """
        h = self.dropout(h)

        attention_outputs = [
            att(h, mch_mask, comp_val, pref_m, pref_m_comp_val)
            for att in self.attentions
        ]
        h_heads = list(map(lambda x: x[0], attention_outputs))
        pref_heads = list(map(lambda x: x[1], attention_outputs))
        if self.concat:
            # h.shape : [sz_b, M, output_dim*num_heads]
            h = torch.cat(h_heads, dim=-1)
            pref = torch.cat(pref_heads, dim=-1)
        else:
            # h.shape : [sz_b, M, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            h = h.mean(dim=-1)
            pref = torch.stack(pref_heads, dim=-1)
            pref = pref.mean(dim=-1)

        return (
            h if self.activation is None else self.activation(h),
            pref if self.activation is None else self.activation(pref),
        )
