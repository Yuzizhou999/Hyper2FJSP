import copy
import time

import hvwfg
import numpy as np
import torch

import train
from common_utils import (
    das_dennis,
    find_pareto_efficient_solutions,
    greedy_select_action,
    strToSuffix,
)
from data_utils import load_data_from_files
from enums import ObjectiveFn
from mo_fjsp_env_same_op_nums import MOFJSPEnvForSameOpNums
from mo_fjsp_env_various_op_nums import MOFJSPEnvForVariousOpNums


class MOTrainer(train.Trainer):
    def __init__(self, config):
        super().__init__(config, multi_objective=True)
        self.device = config.device

    def _set_objective_fn(self, objective_fns: list[str]):
        self.objective_fn = [
            ObjectiveFn(str(obj_fn).lower()) for obj_fn in objective_fns
        ]
        self.num_objectives = len(self.objective_fn)

    def _set_environments(self):
        use_lb_features = getattr(self.config, "use_lb_features", True)

        self.env = MOFJSPEnvForSameOpNums(
            self.n_j,
            self.n_m,
            objective_fns=self.objective_fn,
            data_source=self.data_source,
            use_simple_reward=getattr(self.config, "use_simple_reward", False),
            use_lb_features=use_lb_features,
        )
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data, vali_reference_points = load_data_from_files(
            self.vali_data_path, load_reference_points=True
        )

        if self.data_source == "SD1":
            self.vali_env = MOFJSPEnvForVariousOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
            )
        elif self.data_source == "SD2":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
            )
        elif self.data_source == "JSSP":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
            )
        elif self.data_source == "FFSP":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
            )

        self.num_envs_validation = len(vali_data[0])
        vali_data = (
            list(np.repeat(vali_data[0], self.num_preferences_validation, axis=0)),
            [i for i in vali_data[1] for _ in range(self.num_preferences_validation)],
        )
        self.vali_env.set_initial_data(
            vali_data[0], vali_data[1], deadline_alpha=self.deadline_alpha
        )

        self.vali_reference_points = np.asarray(vali_reference_points)

        if ObjectiveFn.COSTS in self.objective_fn:
            reference_points_cost = np.sum(
                np.max(
                    self.vali_env.true_costs[
                        np.arange(
                            0,
                            self.num_envs_validation * self.num_preferences_validation,
                            self.num_preferences_validation,
                        )
                    ],
                    axis=-1,
                ),
                axis=-1,
            ).data
            self.vali_reference_points = np.concatenate(
                (
                    self.vali_reference_points,
                    reference_points_cost[:, None],
                ),
                axis=1,
            )

        if ObjectiveFn.NEGATIVE_MAKESPAN in self.objective_fn:
            self.vali_reference_points = np.concatenate(
                (
                    self.vali_reference_points,
                    np.zeros((self.vali_reference_points.shape[0], 1)),
                ),
                axis=1,
            )

    def _set_model_name(self, config):
        # Create objective suffix if not using the default objectives (total_tardiness and makespan)
        obj_suffix = ""

        # Model name encodes objectives, simple-reward, single-critic, and LB feature usage for quick checkpoint inspection

        # Check if objectives are different from the default pair
        default_objectives = {ObjectiveFn.TOTAL_TARDINESS, ObjectiveFn.MAKESPAN}
        current_objectives = set(self.objective_fn)

        if current_objectives != default_objectives:
            # Create suffix with first three letters of each part in objective name
            obj_abbrs = []
            for obj in self.objective_fn:
                # Get the name from the enum value (e.g. "makespan" from ObjectiveFn.MAKESPAN)
                obj_name = obj.value
                # Split by underscore and take first three characters of each part
                obj_abbr = "_".join([part[:3] for part in obj_name.split("_")])
                obj_abbrs.append(obj_abbr)

            # Join the abbreviations with underscores
            obj_suffix = f"_{'_'.join(obj_abbrs)}"

        single_critic_suffix = "_single_critic" if config.single_value_critic else ""

        # Add simple reward suffix
        simple_reward_suffix = (
            "_SR" if getattr(config, "use_simple_reward", False) else ""
        )

        # Add no-LB-features suffix
        no_lb_suffix = "_NoLB" if not getattr(config, "use_lb_features", True) else ""

        # Create the model name with the objective suffix
        self.model_name = f"{self.data_name}{strToSuffix(config.model_suffix)}_{config.model_architecture.value}{'_no_trans' if config.use_gamma_beta is False else ''}{'_large' if config.hidden_dim_actor > 65 else ''}{obj_suffix}{single_critic_suffix}{simple_reward_suffix}{no_lb_suffix}_MO"

    def _set_initial_rewards(self):
        # Initialize episode rewards to zero
        # We only want to track rewards earned from agent's actions during the episode
        # The initial quality baseline (-init_quality) should not be included in printed rewards
        # as it represents arbitrary starting conditions, not rewards earned by the agent

        ep_rewards = np.zeros((self.num_envs, self.num_objectives))

        return ep_rewards

    def _initialize_episode(self, i_update):
        ep_st = time.time()
        # resampling the training data
        if i_update % self.reset_env_timestep == 0:
            dataset_job_length, dataset_op_pt = self.sample_training_instances()
            state = self.env.set_initial_data(
                dataset_job_length, dataset_op_pt, deadline_alpha=self.deadline_alpha
            )
        else:
            state = self.env.reset()

        # Generate new objective weights for this episode
        objective_weights = np.random.rand(self.num_envs, self.num_objectives)
        self.objective_weights = objective_weights / objective_weights.sum(
            axis=1, keepdims=True
        )
        self.objective_weights_tensor = (
            torch.from_numpy(self.objective_weights).float().to(self.device)
        )

        # For hypernetwork models, assign preferences once per episode
        if hasattr(self.ppo.policy, "assign_preferences"):
            self.ppo.policy.assign_preferences(self.objective_weights_tensor)
            self.ppo.policy_old.assign_preferences(self.objective_weights_tensor)

        return state, ep_st

    def _forward_pass(self, state):
        # For hypernetwork models, use pre-generated parameters (preferences=None)
        # For other models, pass preferences normally
        prefs = (
            None
            if hasattr(self.ppo.policy, "assign_preferences")
            else self.objective_weights_tensor
        )

        pi_envs, vals_envs = self.ppo.policy_old(
            fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
            op_mask=state.op_mask_tensor,  # [sz_b, N, N]
            candidate=state.candidate_tensor,  # [sz_b, J]
            fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
            mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
            comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
            fea_pairs=state.fea_pairs_tensor,  # [sz_b, J, M]
            preferences=prefs,
        )
        return pi_envs, vals_envs

    def _process_reward(self, reward):
        # Use weighted sum
        reward = (reward * self.objective_weights).sum(axis=1)
        return reward

    def _call_update_ppo(self):
        if self.single_value_critic:
            loss, v_loss = self.ppo.update(
                self.memory,
                self.objective_weights_tensor,
                self.single_value_critic,
            )
        else:
            loss, v_loss = self.ppo.update(self.memory, self.objective_weights_tensor)
        return loss, v_loss

    def validate_envs_with_same_op_nums(self, objective_fn):
        self.ppo.policy.eval()
        state = self.vali_env.reset()
        if len(objective_fn) == 2:
            eval_preferences = np.linspace(0, 1, self.num_preferences_validation)
            eval_preferences = np.array([eval_preferences, 1 - eval_preferences]).T
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        elif len(objective_fn) == 3:
            if self.num_preferences_validation == 21:
                eval_preferences = das_dennis(5, 3)  # 21
            elif self.num_preferences_validation == 28:
                eval_preferences = das_dennis(6, 3)
            elif self.num_preferences_validation == 36:
                eval_preferences = das_dennis(7, 3)
            elif self.num_preferences_validation == 45:
                eval_preferences = das_dennis(8, 3)
            else:
                raise NotImplementedError(
                    f"Unsupported number of preferences: {self.num_preferences_validation}"
                )
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        elif len(objective_fn) == 4:
            if self.num_preferences_validation == 15:
                eval_preferences = das_dennis(2, 4)  # 15
            elif self.num_preferences_validation == 20:
                eval_preferences = das_dennis(3, 4)  # 20
            elif self.num_preferences_validation == 35:
                eval_preferences = das_dennis(4, 4)  # 35
            elif self.num_preferences_validation == 56:
                eval_preferences = das_dennis(5, 4)  # 56
            elif self.num_preferences_validation == 84:
                eval_preferences = das_dennis(6, 4)  # 84
            elif self.num_preferences_validation == 120:
                eval_preferences = das_dennis(7, 4)  # 120
            else:
                raise NotImplementedError(
                    f"Unsupported number of preferences: {self.num_preferences_validation}"
                )
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        else:
            raise NotImplementedError("Unsupported number of objectives.")

        # For HYPER_DANIEL: pre-generate actor parameters for all validation preferences
        if hasattr(self.ppo.policy, "assign_preferences"):
            # Repeat preferences to match batch size: [num_envs * num_prefs, pref_dim]
            repeated_preferences = eval_preferences.repeat(self.num_envs_validation, 1)
            self.ppo.policy.assign_preferences(repeated_preferences)

        while True:
            with torch.no_grad():
                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,  # [sz_b, J]
                    fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                    mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                    comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                    fea_pairs=state.fea_pairs_tensor,
                    preferences=None
                    if hasattr(self.ppo.policy, "assign_preferences")
                    else eval_preferences.repeat(
                        state.fea_j_tensor.shape[0] // eval_preferences.shape[0], 1
                    ),
                )  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        eval_objectives = []
        reference_point_indices = []
        lb_indices = []  # Track which index in objective_lower_bounds corresponds to each eval_objective

        if ObjectiveFn.MAKESPAN in objective_fn:
            eval_objectives.append(self.vali_env.current_makespan)
            reference_point_indices.append(0)
            lb_indices.append(self.vali_env.objectives.index(ObjectiveFn.MAKESPAN))
        if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn:
            eval_objectives.append(self.vali_env.compute_avg_flowtime())
            reference_point_indices.append(1)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.AVERAGE_FLOWTIME)
            )
        if ObjectiveFn.TOTAL_TARDINESS in objective_fn:
            eval_objectives.append(self.vali_env.compute_total_tardiness())
            reference_point_indices.append(2)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.TOTAL_TARDINESS)
            )
        if ObjectiveFn.TOTAL_EARLINESS in objective_fn:
            eval_objectives.append(self.vali_env.compute_total_earliness())
            # Map deadline_alpha to appropriate earliness reference point index
            # Index 3: alpha=0.5, Index 4: alpha=0.9, Index 5: alpha=1.0
            if self.deadline_alpha <= 0.51:
                reference_point_indices.append(3)  # alpha 0.5
            elif self.deadline_alpha <= 0.91:
                reference_point_indices.append(4)  # alpha 0.9
            else:
                reference_point_indices.append(5)  # alpha 1.0
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.TOTAL_EARLINESS)
            )
        if ObjectiveFn.COSTS in objective_fn:
            eval_objectives.append(self.vali_env.compute_costs())
            reference_point_indices.append(len(self.vali_reference_points[0]) - 1)
            lb_indices.append(self.vali_env.objectives.index(ObjectiveFn.COSTS))
        normalization_values = copy.deepcopy(reference_point_indices)
        if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
            eval_objectives.append(-self.vali_env.current_makespan)
            reference_point_indices.append(-1)
            normalization_values.append(0)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN)
            )
        eval_objectives = np.stack(
            eval_objectives, axis=1
        )
        eval_objectives = eval_objectives.reshape(
            self.num_envs_validation, self.num_preferences_validation, -1
        )
        normalized_hypervolumes = []
        for i in range(eval_objectives.shape[0]):
            # Reorder lower_bounds to match eval_objectives ordering
            all_lower_bounds = self.vali_env.objective_lower_bounds[
                i * self.num_preferences_validation
            ]
            lower_bounds = all_lower_bounds[lb_indices]
            normalized_hypervolume = hvwfg.wfg(
                find_pareto_efficient_solutions(eval_objectives[i]) - lower_bounds,
                self.vali_reference_points[i][reference_point_indices] - lower_bounds,
            ) / np.prod(
                self.vali_reference_points[i][normalization_values] - lower_bounds
            )
            normalized_hypervolumes.append(-normalized_hypervolume)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        return normalized_hypervolumes

    def validate_envs_with_various_op_nums(self, objective_fn):
        self.ppo.policy.eval()
        state = self.vali_env.reset()
        if len(objective_fn) == 2:
            eval_preferences = np.linspace(0, 1, self.num_preferences_validation)
            eval_preferences = np.array([eval_preferences, 1 - eval_preferences]).T
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        elif len(objective_fn) == 3:
            if self.num_preferences_validation == 21:
                eval_preferences = das_dennis(5, 3)  # 21
            elif self.num_preferences_validation == 28:
                eval_preferences = das_dennis(6, 3)
            elif self.num_preferences_validation == 36:
                eval_preferences = das_dennis(7, 3)
            elif self.num_preferences_validation == 45:
                eval_preferences = das_dennis(8, 3)
            else:
                raise NotImplementedError(
                    f"Unsupported number of preferences: {self.num_preferences_validation}"
                )
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        elif len(objective_fn) == 4:
            if self.num_preferences_validation == 15:
                eval_preferences = das_dennis(2, 4)  # 15
            elif self.num_preferences_validation == 20:
                eval_preferences = das_dennis(3, 4)  # 20
            elif self.num_preferences_validation == 35:
                eval_preferences = das_dennis(4, 4)  # 35
            elif self.num_preferences_validation == 56:
                eval_preferences = das_dennis(5, 4)  # 56
            elif self.num_preferences_validation == 84:
                eval_preferences = das_dennis(6, 4)  # 84
            elif self.num_preferences_validation == 120:
                eval_preferences = das_dennis(7, 4)  # 120
            else:
                raise NotImplementedError(
                    f"Unsupported number of preferences: {self.num_preferences_validation}"
                )
            eval_preferences = (
                torch.from_numpy(eval_preferences).float().to(self.device)
            )
        else:
            raise NotImplementedError("Unsupported number of objectives.")

        # For HYPER_DANIEL: generate parameters once per unique preference for memory efficiency
        if hasattr(self.ppo.policy, "assign_preferences"):
            # Generate parameters for all unique preferences (e.g., 40 instead of 400)
            self.ppo.policy.assign_preferences(eval_preferences)
            use_preference_indices = True
        else:
            use_preference_indices = False

        while True:
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)

                if use_preference_indices:
                    # Map each active environment to its preference index
                    # Environments are ordered: [env0_pref0, env0_pref1, ..., env1_pref0, ...]
                    active_indices = torch.where(batch_idx)[0]
                    # Each environment's preference index is: env_idx % num_preferences
                    pref_indices = active_indices % self.num_preferences_validation

                    pi, _ = self.ppo.policy(
                        fea_j=state.fea_j_tensor[batch_idx],
                        op_mask=state.op_mask_tensor[batch_idx],
                        candidate=state.candidate_tensor[batch_idx],
                        fea_m=state.fea_m_tensor[batch_idx],
                        mch_mask=state.mch_mask_tensor[batch_idx],
                        comp_idx=state.comp_idx_tensor[batch_idx],
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                        fea_pairs=state.fea_pairs_tensor[batch_idx],
                        preferences=None,  # Already assigned
                        preference_indices=pref_indices,  # Map to unique preferences
                    )
                else:
                    # Non-hypernetwork models: process normally
                    pi, _ = self.ppo.policy(
                        fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                        op_mask=state.op_mask_tensor[batch_idx],
                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                            batch_idx
                        ],  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor[batch_idx],
                        preferences=eval_preferences.repeat(
                            state.fea_j_tensor[batch_idx].shape[0]
                            // eval_preferences.shape[0],
                            1,
                        ),
                    )  # [sz_b, J, M]
            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        eval_objectives = []
        reference_point_indices = []
        lb_indices = []  # Track which index in objective_lower_bounds corresponds to each eval_objective

        if ObjectiveFn.MAKESPAN in objective_fn:
            eval_objectives.append(self.vali_env.current_makespan)
            reference_point_indices.append(0)
            lb_indices.append(self.vali_env.objectives.index(ObjectiveFn.MAKESPAN))
        if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn:
            eval_objectives.append(self.vali_env.compute_avg_flowtime())
            reference_point_indices.append(1)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.AVERAGE_FLOWTIME)
            )
        if ObjectiveFn.TOTAL_TARDINESS in objective_fn:
            eval_objectives.append(self.vali_env.compute_total_tardiness())
            reference_point_indices.append(2)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.TOTAL_TARDINESS)
            )
        if ObjectiveFn.TOTAL_EARLINESS in objective_fn:
            eval_objectives.append(self.vali_env.compute_total_earliness())
            # Map deadline_alpha to appropriate earliness reference point index
            # Index 3: alpha=0.5, Index 4: alpha=0.9, Index 5: alpha=1.0
            if self.deadline_alpha <= 0.51:
                reference_point_indices.append(3)  # alpha 0.5
            elif self.deadline_alpha <= 0.91:
                reference_point_indices.append(4)  # alpha 0.9
            else:
                reference_point_indices.append(5)  # alpha 1.0
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.TOTAL_EARLINESS)
            )
        if ObjectiveFn.COSTS in objective_fn:
            eval_objectives.append(self.vali_env.compute_costs())
            reference_point_indices.append(len(self.vali_reference_points[0]) - 1)
            lb_indices.append(self.vali_env.objectives.index(ObjectiveFn.COSTS))
        normalization_values = copy.deepcopy(reference_point_indices)
        if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
            eval_objectives.append(-self.vali_env.current_makespan)
            reference_point_indices.append(-1)
            normalization_values.append(0)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN)
            )
        eval_objectives = np.stack(eval_objectives, axis=1)
        eval_objectives = eval_objectives.reshape(
            self.num_envs_validation, self.num_preferences_validation, -1
        )
        normalized_hypervolumes = []
        for i in range(eval_objectives.shape[0]):
            # Reorder lower_bounds to match eval_objectives ordering
            all_lower_bounds = self.vali_env.objective_lower_bounds[
                i * self.num_preferences_validation
            ]
            lower_bounds = all_lower_bounds[lb_indices]
            normalized_hypervolume = hvwfg.wfg(
                find_pareto_efficient_solutions(eval_objectives[i]) - lower_bounds,
                self.vali_reference_points[i][reference_point_indices] - lower_bounds,
            ) / np.prod(
                self.vali_reference_points[i][normalization_values] - lower_bounds
            )
            normalized_hypervolumes.append(-normalized_hypervolume)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        return normalized_hypervolumes


def main():
    trainer = MOTrainer(train.configurations)
    trainer.train()


if __name__ == "__main__":
    main()
