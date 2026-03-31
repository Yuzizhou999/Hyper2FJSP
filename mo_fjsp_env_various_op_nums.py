import copy
import numpy as np
import numpy.ma as ma

from dynamic_event_engine import DynamicEventEngine
from enums import ObjectiveFn
from fjsp_env_various_op_nums import EnvState, FJSPEnvForVariousOpNums
from multipliers import MULTIPLIERS


class MOFJSPEnvForVariousOpNums(FJSPEnvForVariousOpNums):
    def __init__(
        self,
        n_j,
        n_m,
        objective_fns: list[ObjectiveFn],
        data_source: str,
        use_simple_reward: bool = False,
        use_lb_features: bool = True,
        dynamic_events_enabled: bool = False,
        dynamic_event_prob: float = 0.0,
        dynamic_event_types: list[str] | None = None,
        dynamic_breakdown_duration: int = 2,
        dynamic_deadline_shift_scale: float = 0.1,
        dynamic_energy_duration: int = 2,
        dynamic_energy_scale: float = 0.2,
        dynamic_seed: int | None = None,
    ):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()
        self.data_source = data_source

        # Simple actual-value reward strategy (training-only option)
        self.use_simple_reward = use_simple_reward

        # Lower bound features flag
        self.use_lb_features = use_lb_features

        # the dimension of operation raw features
        # Base: 9 features (scheduled, min_pt, pt_span, mean_pt, waiting, remain_work, left_ops, job_remain_work, available_mchs)
        self.op_fea_dim = 9

        # Add op_ct_lb if using lower bound features
        if use_lb_features:
            self.op_fea_dim += 1  # op_ct_lb

        # Add objective-specific LB features only if use_lb_features is True
        # op_lateness_lb is used for both tardiness and earliness objectives
        if (
            ObjectiveFn.TOTAL_TARDINESS in objective_fns
            or ObjectiveFn.TOTAL_EARLINESS in objective_fns
        ):
            if use_lb_features:
                self.op_fea_dim += 1  # op_lateness_lb

        if ObjectiveFn.AVERAGE_FLOWTIME in objective_fns:
            if use_lb_features:
                self.op_fea_dim += 1  # op_flowtime_lb

        self.mch_fea_dim = 8
        self.objectives = objective_fns
        self.nr_objectives = len(objective_fns)

        # Use SD1_SR multipliers when simple reward mode is enabled
        if self.use_simple_reward:
            data_source_key = "SD1_SR"
        else:
            data_source_key = self.data_source

        try:
            self.multipliers = MULTIPLIERS[
                (data_source_key, self.number_of_jobs, self.number_of_machines)
            ]
        except KeyError:
            Warning(
                f"Multipliers for {data_source_key} with {self.number_of_jobs} jobs and {self.number_of_machines} machines not found. Using default multipliers."
            )
            self.multipliers = MULTIPLIERS["BenchData"]

        self.dynamic_event_engine = DynamicEventEngine(
            enabled=dynamic_events_enabled,
            event_prob=dynamic_event_prob,
            event_types=dynamic_event_types,
            breakdown_duration=dynamic_breakdown_duration,
            deadline_shift_scale=dynamic_deadline_shift_scale,
            energy_duration=dynamic_energy_duration,
            energy_scale=dynamic_energy_scale,
            seed=dynamic_seed,
        )
        self._base_costs_data = None
        self._base_true_costs_data = None

    def _sync_runtime_arrays_from_state(self):
        """
        Keep runtime numpy arrays consistent with the current state snapshot.

        Base reset() restores self.state, but runtime arrays used by dynamic
        event hooks are not all reassigned there.
        """
        self.fea_j = self.state.fea_j_tensor.detach().cpu().numpy()
        self.op_mask = self.state.op_mask_tensor.detach().cpu().numpy()
        self.fea_m = self.state.fea_m_tensor.detach().cpu().numpy()
        self.mch_mask = self.state.mch_mask_tensor.detach().cpu().numpy()
        self.dynamic_pair_mask = (
            self.state.dynamic_pair_mask_tensor.detach().cpu().numpy().astype(bool)
        )
        self.comp_idx = self.state.comp_idx_tensor.detach().cpu().numpy()
        self.candidate = self.state.candidate_tensor.detach().cpu().numpy()
        self.fea_pairs = self.state.fea_pairs_tensor.detach().cpu().numpy()

    def set_initial_data(self, job_length_list, op_pt_list, deadline_alpha=1.0):
        state = super().set_initial_data(job_length_list, op_pt_list, deadline_alpha)
        self._base_costs_data = np.copy(self.costs.data)
        self._base_true_costs_data = np.copy(self.true_costs.data)
        self.dynamic_event_engine.reset(
            num_envs=self.number_of_envs,
            num_jobs=self.number_of_jobs,
            num_machines=self.number_of_machines,
        )
        self._apply_dynamic_events_after_transition(
            done_mask=np.zeros(self.number_of_envs, dtype=bool),
            apply_sampling=False,
        )
        self._refresh_state_with_event_context()
        self.old_state = copy.deepcopy(self.state)
        return self.state

    def reset(self):
        state = super().reset()
        self._sync_runtime_arrays_from_state()
        self.dynamic_event_engine.reset(
            num_envs=self.number_of_envs,
            num_jobs=self.number_of_jobs,
            num_machines=self.number_of_machines,
        )
        self._apply_dynamic_events_after_transition(
            done_mask=np.zeros(self.number_of_envs, dtype=bool),
            apply_sampling=False,
        )
        self._refresh_state_with_event_context()
        self.old_state = copy.deepcopy(self.state)
        return self.state

    def step(self, actions):
        state, reward, done = super().step(actions)
        done_mask = np.asarray(done).astype(bool)
        self._apply_dynamic_events_after_transition(done_mask=done_mask, apply_sampling=True)
        self._refresh_state_with_event_context()
        return self.state, reward, done

    def _apply_dynamic_events_after_transition(self, done_mask, apply_sampling: bool):
        if apply_sampling:
            snapshot = self.dynamic_event_engine.sample(
                current_deadlines=self.deadlines,
                done_mask=done_mask,
            )
        else:
            snapshot = self.dynamic_event_engine.sample(
                current_deadlines=self.deadlines,
                done_mask=np.ones_like(done_mask, dtype=bool),
            )

        if snapshot.deadline_delta.shape == self.deadlines.shape:
            self.deadlines = np.maximum(self.deadlines + snapshot.deadline_delta, 1e-8)
            self.true_deadlines = (
                self.deadlines * (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
                + self.pt_lower_bound
            )
            self.compute_tardiness_features()
            self.compute_flowtime_features()
            self.construct_op_features()

        if (
            self._base_costs_data is not None
            and snapshot.energy_multiplier.shape[0] == self.number_of_envs
        ):
            self.cost_lb = self.cost_lb.astype(np.float64, copy=False)
            self.true_cost_lb = self.true_cost_lb.astype(np.float64, copy=False)
            old_min_cost = ma.min(self.costs, axis=-1).filled(0.0)
            old_min_true_cost = ma.min(self.true_costs, axis=-1).filled(0.0)
            energy = snapshot.energy_multiplier[:, None, :]
            self.costs = ma.array(
                self._base_costs_data * energy,
                mask=self.reverse_process_relation,
            )
            self.true_costs = ma.array(
                self._base_true_costs_data * energy,
                mask=self.reverse_process_relation,
            )
            new_min_cost = ma.min(self.costs, axis=-1).filled(0.0)
            new_min_true_cost = ma.min(self.true_costs, axis=-1).filled(0.0)
            unscheduled_mask = (self.op_scheduled_flag == 0).astype(np.float64)
            self.cost_lb += np.sum(
                (new_min_cost - old_min_cost) * unscheduled_mask, axis=1
            )
            self.true_cost_lb += np.sum(
                (new_min_true_cost - old_min_true_cost) * unscheduled_mask, axis=1
            )

        prev_dynamic_pair_mask = np.copy(self.dynamic_pair_mask)
        breakdown_mask = snapshot.breakdown_mask
        if breakdown_mask.shape == (self.number_of_envs, self.number_of_machines):
            self.dynamic_pair_mask = np.logical_or(
                self.dynamic_pair_mask, breakdown_mask[:, None, :]
            )
            invalid_envs = self.dynamic_pair_mask.reshape(self.number_of_envs, -1).all(
                axis=1
            )
            self.dynamic_pair_mask[invalid_envs] = prev_dynamic_pair_mask[invalid_envs]

            self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
            self.update_mch_mask()
            self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
            self.construct_mch_features()
            self.construct_pair_features()

    def _refresh_state_with_event_context(self):
        event_context = self.dynamic_event_engine.build_global_context()
        event_mch = self.dynamic_event_engine.build_machine_context()
        self.state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
            event_context=event_context,
            event_mch=event_mch,
        )

    def construct_op_features(self):
        if self.use_lb_features:
            # Include all features including LB features
            self.fea_j = np.stack(
                (
                    self.op_scheduled_flag,
                    self.op_ct_lb,  # LB feature
                    self.op_min_pt,
                    self.pt_span,
                    self.op_mean_pt,
                    self.op_waiting_time,
                    self.op_remain_work,
                    self.op_match_job_left_op_nums,
                    self.op_match_job_remain_work,
                    self.op_available_mch_nums,
                ),
                axis=2,
            )
        else:
            # Without LB features (exclude op_ct_lb only)
            self.fea_j = np.stack(
                (
                    self.op_scheduled_flag,
                    # op_ct_lb excluded
                    self.op_min_pt,
                    self.pt_span,
                    self.op_mean_pt,
                    self.op_waiting_time,
                    self.op_remain_work,
                    self.op_match_job_left_op_nums,
                    self.op_match_job_remain_work,
                    self.op_available_mch_nums,
                ),
                axis=2,
            )

        # Add objective-specific LB features only if use_lb_features is True
        # op_lateness_lb is used for both tardiness and earliness objectives
        if (
            ObjectiveFn.TOTAL_TARDINESS in self.objectives
            or ObjectiveFn.TOTAL_EARLINESS in self.objectives
        ):
            if self.use_lb_features:
                self.fea_j = np.concatenate(
                    (self.fea_j, self.op_lateness_lb[..., None]), axis=2
                )

        if ObjectiveFn.AVERAGE_FLOWTIME in self.objectives:
            if self.use_lb_features:
                self.fea_j = np.concatenate(
                    (self.fea_j, self.op_flowtime_lb[..., None]), axis=2
                )
        if self.flag_exist_dummy_node:
            mask_all = np.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def compute_tardiness_features(self):
        # op_lateness_lb is used for both tardiness and earliness objectives
        if (
            ObjectiveFn.TOTAL_TARDINESS in self.objectives
            or ObjectiveFn.TOTAL_EARLINESS in self.objectives
        ):
            # Compute the lateness for each job
            job_lateness = (
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.deadlines
            )
            # Build an op->job mapping using job_first_op_id and job_last_op_id
            ops = np.arange(self.number_of_ops)[
                None, None, :
            ]  # shape (1, 1, number_of_ops)
            job_first = self.job_first_op_id[
                ..., None
            ]  # shape (number_of_envs, number_of_jobs, 1)
            job_last = self.job_last_op_id[
                ..., None
            ]  # shape (number_of_envs, number_of_jobs, 1)
            job_mask = (ops >= job_first) & (
                ops <= job_last
            )  # shape (number_of_envs, number_of_jobs, number_of_ops)
            op_job_id = job_mask.argmax(axis=1)  # shape (number_of_envs, number_of_ops)
            self.op_lateness_lb = job_lateness[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]  # shape (number_of_envs, number_of_ops)

    def compute_flowtime_features(self):
        if ObjectiveFn.AVERAGE_FLOWTIME in self.objectives:
            job_flowtime_lb = np.where(
                self.job_start_times == np.inf,
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ],
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.job_start_times,
            )
            ops = np.arange(self.number_of_ops)[None, None, :]
            job_first = self.job_first_op_id[..., None]
            job_last = self.job_last_op_id[..., None]
            job_mask = (ops >= job_first) & (ops <= job_last)
            op_job_id = job_mask.argmax(axis=1)
            self.op_flowtime_lb = job_flowtime_lb[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]

    def set_initial_reward_quality(self):
        # If using simple actual-value reward, initialize all qualities to 0 since nothing is scheduled yet
        if self.use_simple_reward:
            self.init_quality = np.zeros(
                (self.number_of_envs, self.nr_objectives), dtype=np.float64
            )
            # Still compute objective_lower_bounds for reference
            objective_lower_bounds = []
            for obj in self.objectives:
                if obj == ObjectiveFn.MAKESPAN:
                    objective_lower_bounds.append(np.max(self.true_op_ct_lb, axis=1))
                elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
                    objective_lower_bounds.append(np.mean(self.true_op_ct_lb, axis=1))
                elif obj == ObjectiveFn.TOTAL_TARDINESS:
                    true_tardiness_estimates = np.maximum(
                        0,
                        self.true_op_ct_lb[
                            np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                        ]
                        - self.true_deadlines,
                    )
                    objective_lower_bounds.append(
                        np.sum(true_tardiness_estimates, axis=1)
                    )
                elif obj == ObjectiveFn.TOTAL_EARLINESS:
                    objective_lower_bounds.append(np.zeros(self.number_of_envs))
                elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                    objective_lower_bounds.append(-np.max(self.true_op_ct_lb, axis=1))
                elif obj == ObjectiveFn.COSTS:
                    objective_lower_bounds.append(np.copy(self.true_cost_lb))
                else:
                    objective_lower_bounds.append(np.zeros(self.number_of_envs))
            self.objective_lower_bounds = np.stack(objective_lower_bounds, axis=-1).data
            return

        # Original lower-bound based initialization
        qualities = []
        objective_lower_bounds = []
        for obj in self.objectives:
            if obj == ObjectiveFn.MAKESPAN:
                quality = np.max(self.op_ct_lb, axis=1)
                objective_lower_bounds.append(np.max(self.true_op_ct_lb, axis=1))
            elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
                quality = np.mean(self.op_ct_lb, axis=1)
                objective_lower_bounds.append(np.mean(self.true_op_ct_lb, axis=1))
            elif obj == ObjectiveFn.TOTAL_TARDINESS:
                tardiness_estimates = np.maximum(
                    0,
                    self.op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ]
                    - self.deadlines,
                )
                quality = np.sum(tardiness_estimates, axis=1)
                true_tardiness_estimates = np.maximum(
                    0,
                    self.true_op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ]
                    - self.true_deadlines,
                )
                objective_lower_bounds.append(np.sum(true_tardiness_estimates, axis=1))
            elif obj == ObjectiveFn.TOTAL_EARLINESS:
                # Upper bound on earliness using lower bounds
                earliness_ub_estimates = np.maximum(
                    0,
                    self.deadlines
                    - self.op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ],
                )
                quality = np.sum(earliness_ub_estimates, axis=1)
                # Lower bound for earliness is 0 (best case: all jobs finish exactly at deadline)
                objective_lower_bounds.append(np.zeros(self.number_of_envs))
            elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                quality = -np.max(self.op_ct_lb, axis=1)
                objective_lower_bounds.append(-np.max(self.true_op_ct_lb, axis=1))
            elif obj == ObjectiveFn.COSTS:
                quality = np.copy(self.cost_lb)
                objective_lower_bounds.append(np.copy(self.true_cost_lb))
            else:
                quality = np.zeros(self.number_of_envs)
                objective_lower_bounds.append(np.zeros(self.number_of_envs))
            qualities.append(quality)
        # Stack qualities to have one value per instance per objective
        self.init_quality = np.stack(qualities, axis=-1)
        self.objective_lower_bounds = np.stack(objective_lower_bounds, axis=-1).data

    def compute_reward(self):
        if self.use_simple_reward:
            return self._compute_simple_actual_value_reward()

        rewards = []
        # For each objective in self.objectives, compute a quality measure,
        # then compute the reward as the decrease in that quality.
        for i, obj in enumerate(self.objectives):
            if obj == ObjectiveFn.MAKESPAN:
                current_quality = np.max(self.op_ct_lb, axis=1)
            elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                current_quality = -np.max(self.op_ct_lb, axis=1)
            elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
                flowtime_estimates = np.zeros(
                    (self.number_of_envs, self.number_of_jobs)
                )
                for env in range(self.number_of_envs):
                    for job in range(self.number_of_jobs):
                        if self.job_start_times[env, job] == np.inf:
                            flowtime_estimates[env, job] = np.sum(
                                self.op_min_pt[
                                    env,
                                    self.job_first_op_id[
                                        env, job
                                    ] : self.job_last_op_id[env, job] + 1,
                                ]
                            )
                        else:
                            flowtime_estimates[env, job] = (
                                np.max(
                                    self.op_ct_lb[
                                        env,
                                        self.job_first_op_id[
                                            env, job
                                        ] : self.job_last_op_id[env, job] + 1,
                                    ]
                                )
                                - self.job_start_times[env, job]
                            )
                current_quality = np.mean(flowtime_estimates, axis=1)
            elif obj == ObjectiveFn.TOTAL_TARDINESS:
                tardiness_estimates = np.maximum(
                    0,
                    self.op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ]
                    - self.deadlines,
                )
                current_quality = np.sum(tardiness_estimates, axis=1)
            elif obj == ObjectiveFn.TOTAL_EARLINESS:
                # Upper bound on earliness (how early jobs could finish)
                earliness_ub_estimates = np.maximum(
                    0,
                    self.deadlines
                    - self.op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ],
                )
                current_quality = np.sum(earliness_ub_estimates, axis=1)
            elif obj == ObjectiveFn.COSTS:
                current_quality = np.copy(self.cost_lb)
            else:
                current_quality = np.zeros(self.number_of_envs)

            # Compute reward as the improvement (decrease) in the objective quality.
            reward = self.previous_obj_estimate[:, i] - current_quality
            reward *= self.multipliers[obj]
            rewards.append(reward)
            # Update the previous estimate for this objective.
            self.previous_obj_estimate[:, i] = current_quality

        # Stack rewards so that each instance has one reward per objective.
        reward = np.stack(rewards, axis=-1)
        return reward

    def _compute_simple_actual_value_reward(self) -> np.ndarray:
        """
        Compute rewards using actual scheduled values instead of lower bound estimates.

        Rules for computing current quality:
        - Makespan: use self.current_makespan (actual max completion time among scheduled ops)
        - Flowtime: for each job, use last scheduled operation's completion - job start time (0 if no ops scheduled)
        - Tardiness: for each job, use max(0, last scheduled op completion - deadline) (0 if no ops scheduled)
        - Costs: use self.actual_cost_sum (accumulated actual costs of scheduled operations)

        :return: multi-objective rewards [num_envs, num_objectives]
        """
        rewards = []
        for i, obj in enumerate(self.objectives):
            if obj == ObjectiveFn.MAKESPAN:
                # Use actual makespan; treat -inf as 0 for initial state
                current_quality = np.where(
                    np.isneginf(self.current_makespan), 0.0, self.current_makespan
                )
            elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                current_quality = -np.where(
                    np.isneginf(self.current_makespan), 0.0, self.current_makespan
                )
            elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
                # Compute flowtime for each job using only scheduled operations
                flowtime_actual = np.zeros((self.number_of_envs, self.number_of_jobs))
                for env in range(self.number_of_envs):
                    for job in range(self.number_of_jobs):
                        j_first = self.job_first_op_id[env, job]
                        j_last = self.job_last_op_id[env, job]
                        job_ops = np.arange(j_first, j_last + 1)
                        scheduled_mask = self.op_scheduled_flag[env, job_ops] == 1
                        if np.any(scheduled_mask):
                            # Find last scheduled operation for this job
                            last_scheduled_op = job_ops[scheduled_mask][-1]
                            completion = self.true_op_ct[env, last_scheduled_op]
                            if self.true_job_start_times[env, job] != np.inf:
                                flowtime_actual[env, job] = max(
                                    0.0,
                                    completion - self.true_job_start_times[env, job],
                                )
                            else:
                                flowtime_actual[env, job] = 0.0
                        else:
                            flowtime_actual[env, job] = 0.0
                current_quality = np.mean(flowtime_actual, axis=1)
            elif obj == ObjectiveFn.TOTAL_TARDINESS:
                # Compute tardiness for each job using only scheduled operations
                tardiness_actual = np.zeros((self.number_of_envs, self.number_of_jobs))
                for env in range(self.number_of_envs):
                    for job in range(self.number_of_jobs):
                        j_first = self.job_first_op_id[env, job]
                        j_last = self.job_last_op_id[env, job]
                        job_ops = np.arange(j_first, j_last + 1)
                        scheduled_mask = self.op_scheduled_flag[env, job_ops] == 1
                        if np.any(scheduled_mask):
                            # Find last scheduled operation for this job
                            last_scheduled_op = job_ops[scheduled_mask][-1]
                            completion = self.true_op_ct[env, last_scheduled_op]
                            tardiness_actual[env, job] = max(
                                0.0, completion - self.true_deadlines[env, job]
                            )
                        else:
                            tardiness_actual[env, job] = 0.0
                current_quality = np.sum(tardiness_actual, axis=1)
            elif obj == ObjectiveFn.TOTAL_EARLINESS:
                # Compute earliness for each job using actual completion times
                earliness_actual = np.zeros((self.number_of_envs, self.number_of_jobs))
                for env in range(self.number_of_envs):
                    for job in range(self.number_of_jobs):
                        j_first = self.job_first_op_id[env, job]
                        j_last = self.job_last_op_id[env, job]
                        job_ops = np.arange(j_first, j_last + 1)
                        scheduled_mask = self.op_scheduled_flag[env, job_ops] == 1
                        if np.any(scheduled_mask):
                            # Find last scheduled operation completion time
                            last_scheduled_op = job_ops[scheduled_mask][-1]
                            job_completion = self.true_op_ct[env, last_scheduled_op]
                            # Earliness = max(0, deadline - completion)
                            earliness_actual[env, job] = max(
                                0, self.true_deadlines[env, job] - job_completion
                            )
                        else:
                            earliness_actual[env, job] = 0  # No ops scheduled yet
                current_quality = np.sum(earliness_actual, axis=1)
            elif obj == ObjectiveFn.COSTS:
                # Use accumulated actual costs from scheduled operations
                current_quality = np.copy(self.actual_cost_sum)
            else:
                current_quality = np.zeros(self.number_of_envs)

            # Compute reward as improvement (decrease) in objective quality
            reward = self.previous_obj_estimate[:, i] - current_quality
            # Apply SD1_SR multipliers to scale the actual value rewards
            reward *= self.multipliers[obj]
            rewards.append(reward)
            # Update previous estimate for next step
            self.previous_obj_estimate[:, i] = current_quality

        return np.stack(rewards, axis=-1)
