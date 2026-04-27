import copy
import time
from pathlib import Path

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
from hypervolume_utils import compute_hypervolume
from mo_fjsp_env_same_op_nums import MOFJSPEnvForSameOpNums
from mo_fjsp_env_various_op_nums import MOFJSPEnvForVariousOpNums
from model.PPO import build_policy


class MOTrainer(train.Trainer):
    def __init__(self, config):
        super().__init__(config, multi_objective=True)
        self.device = torch.device(config.device)
        self.current_update = 0
        self.corner_distill_teacher_specs = []
        self._maybe_load_resume_model()
        self._setup_corner_distillation()

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

        hyper_mode_suffix = ""
        if config.model_architecture.value == "hyper_daniel":
            hyper_mode_suffix = (
                "_inst"
                if getattr(config, "hyper_use_instance_features", False)
                else "_pref_only"
            )
            if getattr(config, "hyper_condition_encoder", False):
                encoder_bits = []
                if getattr(config, "hyper_encoder_use_film", True):
                    encoder_bits.append("film")
                if getattr(config, "hyper_encoder_use_attention", True):
                    encoder_bits.append("attn")
                hyper_mode_suffix += "_enc_" + (
                    "_".join(encoder_bits) if encoder_bits else "off"
                )

        fixed_pref_suffix = ""
        fixed_pref = getattr(config, "fixed_train_preference", None)
        if fixed_pref is not None:
            fixed_pref = np.asarray(fixed_pref, dtype=np.float32)
            if fixed_pref.size == len(self.objective_fn):
                pref_sum = float(fixed_pref.sum())
                if pref_sum > 0:
                    fixed_pref = fixed_pref / pref_sum
                    pref_bits = "_".join(f"{float(x):g}" for x in fixed_pref)
                    main_obj = self.objective_fn[int(np.argmax(fixed_pref))].value
                    fixed_pref_suffix = f"_expert_{main_obj[:3]}_pref_{pref_bits}"

        corner_distill_suffix = ""
        if getattr(config, "corner_distill_enable", False):
            fmt = lambda x: f"{float(x):g}".replace(".", "p")
            corner_distill_suffix = (
                f"_cdist_t{fmt(getattr(config, 'corner_distill_threshold', 0.95))}"
                f"_l{fmt(getattr(config, 'corner_distill_coef', 0.03))}"
            )

        # Create the model name with the objective suffix
        self.model_name = f"{self.data_name}{strToSuffix(config.model_suffix)}_{config.model_architecture.value}{hyper_mode_suffix}{'_no_trans' if config.use_gamma_beta is False else ''}{'_large' if config.hidden_dim_actor > 65 else ''}{obj_suffix}{fixed_pref_suffix}{corner_distill_suffix}{single_critic_suffix}{simple_reward_suffix}{no_lb_suffix}_MO"

    def _get_fixed_train_preference(self):
        fixed_pref = getattr(self.config, "fixed_train_preference", None)
        if fixed_pref is None:
            return None

        fixed_pref = np.asarray(fixed_pref, dtype=np.float32)
        if fixed_pref.ndim != 1 or fixed_pref.size != self.num_objectives:
            raise ValueError(
                f"--fixed_train_preference must contain {self.num_objectives} weights, got {fixed_pref.tolist()}"
            )
        if np.any(fixed_pref < 0):
            raise ValueError("--fixed_train_preference weights must be non-negative")
        pref_sum = float(fixed_pref.sum())
        if pref_sum <= 0:
            raise ValueError("--fixed_train_preference must have positive sum")
        return fixed_pref / pref_sum

    def _resolve_checkpoint_path(self, model_name, model_source):
        if not model_name:
            raise ValueError("Empty checkpoint name.")

        direct_path = Path(model_name)
        if direct_path.exists():
            return direct_path

        if direct_path.suffix != ".pth":
            direct_path = Path(f"{model_name}.pth")
            if direct_path.exists():
                return direct_path

        source = model_source or self.config.model_source
        checkpoint_name = model_name if str(model_name).endswith(".pth") else f"{model_name}.pth"
        checkpoint_path = Path("trained_network") / source / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def _parse_preference(self, preference_text):
        pieces = str(preference_text).replace(";", ",").split(",")
        values = np.asarray([float(piece) for piece in pieces], dtype=np.float32)
        if values.ndim != 1 or values.size != self.num_objectives:
            raise ValueError(
                f"Preference '{preference_text}' must contain {self.num_objectives} values."
            )
        if np.any(values < 0):
            raise ValueError(f"Preference '{preference_text}' contains negative values.")
        pref_sum = float(values.sum())
        if pref_sum <= 0:
            raise ValueError(f"Preference '{preference_text}' has non-positive sum.")
        return values / pref_sum

    def _maybe_load_resume_model(self):
        resume_model_name = getattr(self.config, "resume_model_name", "")
        if not resume_model_name:
            return

        resume_source = getattr(self.config, "resume_model_source", "") or self.config.model_source
        checkpoint_path = self._resolve_checkpoint_path(resume_model_name, resume_source)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.ppo.policy.load_state_dict(state_dict)
        self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict())
        print(f"Loaded student initialization from {checkpoint_path}")

    def _setup_corner_distillation(self):
        if not getattr(self.config, "corner_distill_enable", False):
            return

        teacher_models = list(getattr(self.config, "corner_distill_teacher_models", []))
        teacher_preferences = list(
            getattr(self.config, "corner_distill_teacher_preferences", [])
        )
        if not teacher_models:
            raise ValueError(
                "--corner_distill_enable requires --corner_distill_teacher_models."
            )
        if len(teacher_models) != len(teacher_preferences):
            raise ValueError(
                "--corner_distill_teacher_models and "
                "--corner_distill_teacher_preferences must have the same length."
            )

        teacher_source = (
            getattr(self.config, "corner_distill_teacher_model_source", "")
            or self.config.model_source
        )
        specs = []
        for model_name, preference_text in zip(teacher_models, teacher_preferences):
            preference_np = self._parse_preference(preference_text)
            checkpoint_path = self._resolve_checkpoint_path(model_name, teacher_source)
            teacher_policy = build_policy(self.config, multi_objective=True)
            teacher_policy.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            teacher_policy.eval()
            for param in teacher_policy.parameters():
                param.requires_grad_(False)

            objective_index = int(np.argmax(preference_np))
            specs.append(
                {
                    "policy": teacher_policy,
                    "preference": torch.from_numpy(preference_np)
                    .float()
                    .to(self.device),
                    "preference_np": preference_np,
                    "objective_index": objective_index,
                    "model_name": model_name,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

        self.corner_distill_teacher_specs = specs
        self.ppo.set_corner_distillation(
            specs,
            coef=getattr(self.config, "corner_distill_coef", 0.03),
            threshold=getattr(self.config, "corner_distill_threshold", 0.95),
            gate_power=getattr(self.config, "corner_distill_gate_power", 2.0),
            temperature=getattr(self.config, "corner_distill_temperature", 1.0),
        )
        print(
            "Loaded corner distillation teachers: "
            + ", ".join(
                f"{spec['model_name']}@{spec['preference_np'].tolist()}"
                for spec in specs
            )
        )

    def _sample_random_preferences(self):
        objective_weights = np.random.rand(self.num_envs, self.num_objectives)
        objective_weights = objective_weights / objective_weights.sum(
            axis=1, keepdims=True
        )
        return objective_weights.astype(np.float32)

    def _sample_corner_biased_preferences(self):
        objective_weights = self._sample_random_preferences()
        if (
            not getattr(self.config, "corner_distill_enable", False)
            or not self.corner_distill_teacher_specs
        ):
            return objective_weights

        bias_ratio = float(
            np.clip(getattr(self.config, "corner_distill_preference_bias_ratio", 0.0), 0.0, 1.0)
        )
        num_biased = int(round(self.num_envs * bias_ratio))
        if num_biased <= 0:
            return objective_weights

        min_weight = float(
            np.clip(
                getattr(self.config, "corner_distill_preference_min_weight", 0.95),
                0.0,
                1.0,
            )
        )
        exact_ratio = float(
            np.clip(getattr(self.config, "corner_distill_exact_corner_ratio", 0.25), 0.0, 1.0)
        )
        exact_count = int(round(num_biased * exact_ratio))
        env_indices = np.random.choice(self.num_envs, num_biased, replace=False)

        for rank, env_idx in enumerate(env_indices):
            spec = self.corner_distill_teacher_specs[
                np.random.randint(len(self.corner_distill_teacher_specs))
            ]
            objective_index = spec["objective_index"]
            if rank < exact_count:
                objective_weights[env_idx] = spec["preference_np"]
                continue

            dominant_weight = np.random.uniform(min_weight, 1.0)
            rest = np.random.rand(self.num_objectives).astype(np.float32)
            rest[objective_index] = 0.0
            rest_sum = float(rest.sum())
            if rest_sum <= 0:
                objective_weights[env_idx] = spec["preference_np"]
                continue

            weights = rest / rest_sum * (1.0 - dominant_weight)
            weights[objective_index] = dominant_weight
            objective_weights[env_idx] = weights

        return objective_weights.astype(np.float32)

    def _corner_distill_scale(self):
        if not getattr(self.config, "corner_distill_enable", False):
            return 0.0

        start_update = int(getattr(self.config, "corner_distill_start_update", 0))
        if self.current_update < start_update:
            return 0.0

        ramp_updates = int(getattr(self.config, "corner_distill_ramp_updates", 0))
        if ramp_updates <= 0:
            return 1.0

        return min(1.0, (self.current_update - start_update + 1) / ramp_updates)

    def _set_initial_rewards(self):
        # Initialize episode rewards to zero
        # We only want to track rewards earned from agent's actions during the episode
        # The initial quality baseline (-init_quality) should not be included in printed rewards
        # as it represents arbitrary starting conditions, not rewards earned by the agent

        ep_rewards = np.zeros((self.num_envs, self.num_objectives))

        return ep_rewards

    def _initialize_episode(self, i_update):
        self.current_update = i_update
        ep_st = time.time()
        # resampling the training data
        if i_update % self.reset_env_timestep == 0:
            dataset_job_length, dataset_op_pt = self.sample_training_instances()
            state = self.env.set_initial_data(
                dataset_job_length, dataset_op_pt, deadline_alpha=self.deadline_alpha
            )
        else:
            state = self.env.reset()

        # Generate new objective weights for this episode, or keep a fixed
        # one-hot preference when training a corner expert.
        fixed_pref = self._get_fixed_train_preference()
        if fixed_pref is None:
            self.objective_weights = self._sample_corner_biased_preferences()
        else:
            self.objective_weights = np.repeat(
                fixed_pref[None, :], self.num_envs, axis=0
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
                corner_distill_scale=self._corner_distill_scale(),
            )
        else:
            loss, v_loss = self.ppo.update(
                self.memory,
                self.objective_weights_tensor,
                corner_distill_scale=self._corner_distill_scale(),
            )
        return loss, v_loss

    def _validate_fixed_preference_expert(self):
        fixed_pref = self._get_fixed_train_preference()
        if fixed_pref is None:
            return None

        self.ppo.policy.eval()
        state = self.vali_env.reset()
        fixed_pref_tensor = (
            torch.from_numpy(fixed_pref).float().to(self.device).unsqueeze(0)
        )

        while True:
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                batch_size = state.fea_j_tensor[batch_idx].shape[0]
                preferences = fixed_pref_tensor.repeat(batch_size, 1)

                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor[batch_idx],
                    op_mask=state.op_mask_tensor[batch_idx],
                    candidate=state.candidate_tensor[batch_idx],
                    fea_m=state.fea_m_tensor[batch_idx],
                    mch_mask=state.mch_mask_tensor[batch_idx],
                    comp_idx=state.comp_idx_tensor[batch_idx],
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                    fea_pairs=state.fea_pairs_tensor[batch_idx],
                    preferences=preferences,
                )

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        objective = self.objective_fn[int(np.argmax(fixed_pref))]
        if objective == ObjectiveFn.MAKESPAN:
            return self.vali_env.current_makespan
        if objective == ObjectiveFn.AVERAGE_FLOWTIME:
            return self.vali_env.compute_avg_flowtime()
        if objective == ObjectiveFn.TOTAL_TARDINESS:
            return self.vali_env.compute_total_tardiness()
        if objective == ObjectiveFn.TOTAL_EARLINESS:
            return self.vali_env.compute_total_earliness()
        if objective == ObjectiveFn.COSTS:
            return self.vali_env.compute_costs()
        if objective == ObjectiveFn.NEGATIVE_MAKESPAN:
            return -self.vali_env.current_makespan
        raise ValueError(f"Unsupported fixed-preference objective: {objective}")

    def validate_envs_with_same_op_nums(self, objective_fn):
        fixed_pref_result = self._validate_fixed_preference_expert()
        if fixed_pref_result is not None:
            return fixed_pref_result

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
            normalized_hypervolume = compute_hypervolume(
                find_pareto_efficient_solutions(eval_objectives[i]) - lower_bounds,
                self.vali_reference_points[i][reference_point_indices] - lower_bounds,
            ) / np.prod(
                self.vali_reference_points[i][normalization_values] - lower_bounds
            )
            normalized_hypervolumes.append(-normalized_hypervolume)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        return normalized_hypervolumes

    def validate_envs_with_various_op_nums(self, objective_fn):
        fixed_pref_result = self._validate_fixed_preference_expert()
        if fixed_pref_result is not None:
            return fixed_pref_result

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
            normalized_hypervolume = compute_hypervolume(
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
