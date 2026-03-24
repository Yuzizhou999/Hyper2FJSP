"""
Hypernetwork-based multi-objective DANIEL variants.

This module keeps the original HyperActor and extends it with optional
encoder-side conditioning so preferences can affect both the decision head
and part of the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hyper_network import AttentionPooling, HyperActor, HyperEncoderConditioner
from model.sub_layers import Critic


class HYPER_DANIEL(nn.Module):
    def __init__(self, config):
        super(HYPER_DANIEL, self).__init__()
        device = torch.device(config.device)

        self.pair_input_dim = 8
        self.embedding_output_dim = config.layer_fea_output_dim[-1]
        self.pref_dim = (
            len(config.objective_fn) if isinstance(config.objective_fn, list) else 2
        )

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

        from model.main_model import DualAttentionNetwork, MODualAttentionNetworkOpAndMch

        if self.condition_encoder and self.hyper_encoder_use_attention:
            self.feature_exact = MODualAttentionNetworkOpAndMch(
                config, use_gamma_beta=config.use_gamma_beta
            ).to(device)
        else:
            self.feature_exact = DualAttentionNetwork(config).to(device)

        self.attn_pool_j = AttentionPooling(self.embedding_output_dim).to(device)
        self.attn_pool_m = AttentionPooling(self.embedding_output_dim).to(device)

        self.encoder_conditioner = None
        if self.condition_encoder:
            self.encoder_conditioner = HyperEncoderConditioner(
                pref_dim=self.pref_dim,
                fea_j_input_dim=config.fea_j_input_dim,
                fea_m_input_dim=config.fea_m_input_dim,
                hyper_hidden_dim=config.hyper_hidden_dim,
            ).to(device)

        instance_dim = (
            2 * self.embedding_output_dim if self.use_instance_features else 0
        )
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

        self.current_preferences = None
        self.single_value_critic = config.single_value_critic
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1 if config.single_value_critic else self.pref_dim,
        ).to(device)

    def assign_preferences(self, preferences):
        self.current_preferences = preferences
        if self.encoder_conditioner is not None:
            self.encoder_conditioner.assign(preferences)
        if not self.use_instance_features:
            self.actor.assign(preferences)

    def _resolve_preferences(self, preferences, preference_indices):
        if preferences is not None:
            return preferences
        if self.current_preferences is None:
            raise RuntimeError(
                "Call assign_preferences() before forward(), or pass preferences."
            )
        if preference_indices is not None:
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
        if self.encoder_conditioner is None:
            return self.feature_exact(
                fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx
            )

        conditioner_kwargs = {}
        if preferences is not None:
            conditioner_kwargs["preferences"] = preferences
        else:
            conditioner_kwargs["preference_indices"] = preference_indices

        conditioned_fea_j, conditioned_fea_m, pref_j, pref_m = self.encoder_conditioner(
            fea_j, fea_m, **conditioner_kwargs
        )

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
    ):
        batch_prefs = self._resolve_preferences(preferences, preference_indices)

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

        fea_j_global = self.attn_pool_j(fea_j)
        fea_m_global = self.attn_pool_m(fea_m)

        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        fea_j_jc = torch.gather(fea_j, 1, candidate_idx)

        fea_j_jc_serialized = (
            fea_j_jc.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(
            sz_b, M * J, d
        )

        fea_gj_input = fea_j_global.unsqueeze(1).expand_as(fea_j_jc_serialized)
        fea_gm_input = fea_m_global.unsqueeze(1).expand_as(fea_j_jc_serialized)

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

        instance_global = torch.cat([fea_j_global, fea_m_global], dim=-1)

        actor_preference_indices = None
        if self.use_instance_features:
            self.actor.assign(batch_prefs, instance_features=instance_global)
        elif preferences is not None:
            self.actor.assign(batch_prefs)
        elif preference_indices is not None:
            actor_preference_indices = preference_indices

        candidate_scores = self.actor(
            candidate_feature, preference_indices=actor_preference_indices
        ).squeeze(-1)
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float("-inf")
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
