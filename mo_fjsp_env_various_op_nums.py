import numpy as np

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
