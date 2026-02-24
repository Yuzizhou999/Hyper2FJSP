import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of multi layer perceptrons (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        """Run the MLP on input features.

        :param x: input tensor with shape [..., input_dim]
        :return: output tensor with shape [..., output_dim]
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Actor network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, prefences=None):
        """Compute action logits/scores.

        :param x: input tensor with shape [..., input_dim]
        :param prefences: optional preference weights broadcastable to x for weighted outputs
        :return: action scores with shape [..., output_dim] (or [..., 1] when preferences provided)
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            # return self.linears[self.num_layers - 1](h)
            h = self.linears[self.num_layers - 1](h)
            if prefences is not None:
                h = (h * prefences.unsqueeze(1).repeat(1, h.size(1), 1)).sum(
                    dim=-1, keepdim=True
                )
            return h


class Critic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Critic network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Critic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        """Estimate state value.

        :param x: input tensor with shape [..., input_dim]
        :return: value predictions with shape [..., output_dim]
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class HyperActor(nn.Module):
    """
    Hypernetwork-based Actor that generates network parameters from preference vectors.
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
    ):
        """
        :param num_layers: number of layers in the Actor network (should be 3)
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim: output dimension (typically 1 for actor)
        :param pref_dim: dimensionality of preference vector (e.g., 2 for 2 objectives)
        :param hyper_hidden_dim: hidden dimension for hypernetwork (default: 256)
        :param embd_dim: embedding dimension for parameter generation (default: 2)
        """
        super(HyperActor, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pref_dim = pref_dim
        self.embd_dim = embd_dim
        self.activative = torch.tanh

        if num_layers != 3:
            raise ValueError("HyperActor currently only supports 3 layers!")

        # Hypernetwork: generates embeddings from preferences
        # Architecture: pref_dim -> 256 -> 256 -> (embd_dim * 6)
        self.hyper_output_dim = (
            self.embd_dim * 6
        )  # 6 = 3 layers × 2 params (weight, bias)
        self.hyper_fc1 = nn.Linear(pref_dim, hyper_hidden_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_dim, hyper_hidden_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_dim, self.hyper_output_dim, bias=True)

        # Parameter generators: convert embeddings to actual network parameters
        # Layer 1: input_dim -> hidden_dim
        self.hyper_lin1 = nn.Linear(self.embd_dim, input_dim * hidden_dim)
        self.hyper_bias1 = nn.Linear(self.embd_dim, hidden_dim)

        # Layer 2: hidden_dim -> hidden_dim
        self.hyper_lin2 = nn.Linear(self.embd_dim, hidden_dim * hidden_dim)
        self.hyper_bias2 = nn.Linear(self.embd_dim, hidden_dim)

        # Layer 3: hidden_dim -> output_dim
        self.hyper_lin3 = nn.Linear(self.embd_dim, hidden_dim * output_dim)
        self.hyper_bias3 = nn.Linear(self.embd_dim, output_dim)

        # Storage for generated parameters (set by assign method)
        self.dec_lin1_para = None
        self.dec_bias1_para = None
        self.dec_lin2_para = None
        self.dec_bias2_para = None
        self.dec_lin3_para = None
        self.dec_bias3_para = None

    def assign(self, pref):
        """
        Generate network parameters from preference vector(s).
        Supports both single preference and batched preferences.

        :param pref: preference vector(s)
                    - Single: [pref_dim]
                    - Batched: [batch_size, pref_dim]
        :return: batch_size (1 if single preference, batch_size if batched)
        """
        # Ensure pref has batch dimension
        if pref.dim() == 1:
            pref = pref.unsqueeze(0)  # [pref_dim] -> [1, pref_dim]

        batch_size = pref.shape[0]

        # Generate embeddings through hypernetwork for all preferences in batch
        # pref: [batch_size, pref_dim]
        mid_embd = self.hyper_fc3(self.hyper_fc2(self.hyper_fc1(pref)))
        # mid_embd: [batch_size, embd_dim * 6]

        # Split into 6 chunks of size embd_dim for each batch element
        mid_embd = mid_embd.chunk(6, dim=-1)  # 6 tensors of [batch_size, embd_dim]

        # Generate parameters for each layer
        # Layer 1: input_dim -> hidden_dim
        self.dec_lin1_para = self.hyper_lin1(mid_embd[0]).reshape(
            batch_size, self.hidden_dim, self.input_dim
        )  # [batch_size, hidden_dim, input_dim]
        self.dec_bias1_para = self.hyper_bias1(mid_embd[1]).reshape(
            batch_size, self.hidden_dim
        )  # [batch_size, hidden_dim]

        # Layer 2: hidden_dim -> hidden_dim
        self.dec_lin2_para = self.hyper_lin2(mid_embd[2]).reshape(
            batch_size, self.hidden_dim, self.hidden_dim
        )  # [batch_size, hidden_dim, hidden_dim]
        self.dec_bias2_para = self.hyper_bias2(mid_embd[3]).reshape(
            batch_size, self.hidden_dim
        )  # [batch_size, hidden_dim]

        # Layer 3: hidden_dim -> output_dim
        self.dec_lin3_para = self.hyper_lin3(mid_embd[4]).reshape(
            batch_size, self.output_dim, self.hidden_dim
        )  # [batch_size, output_dim, hidden_dim]
        self.dec_bias3_para = self.hyper_bias3(mid_embd[5]).reshape(
            batch_size, self.output_dim
        )  # [batch_size, output_dim]

        return batch_size

    def forward(self, x, preferences=None, preference_indices=None):
        """
        Forward pass using generated parameters.
        Each sample in the batch uses its corresponding generated parameters.

        :param x: input features [batch_size, seq_len, input_dim]
        :param preferences: preference vectors (optional, for regenerating params on-the-fly)
                           [batch_size, pref_dim] or None
        :param preference_indices: indices mapping each sample to its preference in the parameter set
                                   [batch_size], where each value is in range [0, num_unique_prefs)
                                   If provided, uses indexed parameter selection for memory efficiency
        :return: output scores [batch_size, seq_len, output_dim]
        """
        # If preferences provided, regenerate parameters
        if preferences is not None:
            self.assign(preferences)

        if self.dec_lin1_para is None:
            raise RuntimeError(
                "Must call assign(pref) before forward pass or provide preferences!"
            )

        # Apply 3-layer network with generated parameters
        # For batched parameters, we need to apply each sample's parameters to its corresponding input

        # If preference_indices provided, select the right parameters for each sample
        if preference_indices is not None:
            # Select parameters using indices
            # dec_lin1_para: [num_unique_prefs, hidden_dim, input_dim]
            # preference_indices: [batch_size]
            # Result: [batch_size, hidden_dim, input_dim]
            lin1_para = self.dec_lin1_para[preference_indices]
            bias1_para = self.dec_bias1_para[preference_indices]
            lin2_para = self.dec_lin2_para[preference_indices]
            bias2_para = self.dec_bias2_para[preference_indices]
            lin3_para = self.dec_lin3_para[preference_indices]
            bias3_para = self.dec_bias3_para[preference_indices]
        else:
            # Use parameters directly (standard batched approach)
            lin1_para = self.dec_lin1_para
            bias1_para = self.dec_bias1_para
            lin2_para = self.dec_lin2_para
            bias2_para = self.dec_bias2_para
            lin3_para = self.dec_lin3_para
            bias3_para = self.dec_bias3_para

        # Layer 1: tanh activation
        # We use batched matrix multiplication: bmm for the linear operation
        # x: [batch_size, seq_len, input_dim]
        # lin1_para: [batch_size, hidden_dim, input_dim]
        # We need: [batch_size, seq_len, input_dim] @ [batch_size, input_dim, hidden_dim]

        h = torch.bmm(x, lin1_para.transpose(1, 2))  # [batch_size, seq_len, hidden_dim]
        h = h + bias1_para.unsqueeze(1)  # Add bias: [batch_size, 1, hidden_dim]
        h = torch.tanh(h)

        # Layer 2: tanh activation
        # h: [batch_size, seq_len, hidden_dim]
        # lin2_para: [batch_size, hidden_dim, hidden_dim]
        h = torch.bmm(h, lin2_para.transpose(1, 2))  # [batch_size, seq_len, hidden_dim]
        h = h + bias2_para.unsqueeze(1)  # Add bias: [batch_size, 1, hidden_dim]
        h = torch.tanh(h)

        # Layer 3: no activation (linear output)
        # h: [batch_size, seq_len, hidden_dim]
        # lin3_para: [batch_size, output_dim, hidden_dim]
        h = torch.bmm(h, lin3_para.transpose(1, 2))  # [batch_size, seq_len, output_dim]
        h = h + bias3_para.unsqueeze(1)  # Add bias: [batch_size, 1, output_dim]

        return h
