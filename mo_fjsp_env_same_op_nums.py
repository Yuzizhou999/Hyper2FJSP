import copy
from typing import List

import numpy as np
import numpy.ma as ma

from dynamic_event_engine import DynamicEventEngine
from enums import ObjectiveFn
from fjsp_env_same_op_nums import EnvState, FJSPEnvForSameOpNums
from multipliers import MULTIPLIERS


class MOFJSPEnvForSameOpNums(FJSPEnvForSameOpNums):
    def __init__(
        self,
        n_j: int,
        n_m: int,
        objective_fns: List[ObjectiveFn],
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
        """
        适用于工序数相同实例的多目标动态柔性作业车间调度（MO-FJSP）环境环境
        
        :param n_j: 实例的工件总数
        :param n_m: 实例被提供的机器总数
        :param objective_fns: 被选定用作评估和强化的目标函数列表 (如最小化最大完工时间等)
        :param data_source: 算例的数据来源标识，用于选取特定的归一化乘子
        :param use_simple_reward: 是否采用简化版的真实数值奖励 (仅供特定训练测试使用)
        :param use_lb_features: 是否在节点特征中启用下界 (Lower Bound) 类特征信号 (例如 op_ct_lb, op_lateness_lb 等)
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()
        self.data_source = data_source

        self.objective_weights = None

        # 简化版实际价值奖励策略的标识位
        self.use_simple_reward = use_simple_reward

        # 下界特征激活标识
        self.use_lb_features = use_lb_features

        # 工序原始特征的维度
        # 基础特征共有9个 (scheduled, min_pt, pt_span, mean_pt, waiting, remain_work, left_ops, job_remain_work, available_mchs)
        self.op_fea_dim: int = 9

        # 如果启用下界特征信号，我们将把完工时间下界 op_ct_lb 加入工序特征
        if use_lb_features:
            self.op_fea_dim += 1  # op_ct_lb

        # 针对特定目标特有的LB特征 (同样受限于是否启用 use_lb_features)
        # 如果存在总延迟或总提早时间目标，我们纳入该延迟的下界特征 op_lateness_lb
        if (
            ObjectiveFn.TOTAL_TARDINESS in objective_fns
            or ObjectiveFn.TOTAL_EARLINESS in objective_fns
        ):
            if use_lb_features:
                self.op_fea_dim += 1  # op_lateness_lb

        # 如果包括平均流程时间目标，则追加特征 op_flowtime_lb
        if ObjectiveFn.AVERAGE_FLOWTIME in objective_fns:
            if use_lb_features:
                self.op_fea_dim += 1  # op_flowtime_lb

        # 机器原始特征的维度定为8
        self.mch_fea_dim = 8
        # 所要优化的总体目标函数清单和规模
        self.objectives = objective_fns
        self.nr_objectives = len(objective_fns)

        # 倘若激活简化版评价奖励机制，尝试使用带"_SR"后缀特征的乘子字典参数
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

        # 构建并挂载能够提供运行期间突发偶发事件上下文的动态事件引擎模块
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
        使得 Numpy 运行时的各项特征数组数据与最后一次的状态镜像 (State Snapshot) 保持同步一致对齐。

        在基类的 reset() 方法中会恢复 self.state 原有指向张量，但这将导致一些作为计算依据特设出来的缓存 NumPy 数组 
        (例如动态配对掩码 dynamic_pair_mask 或 comp_idx ) 与模型调用产生分裂落后。针对这种现象采取显式的底层复位对齐。
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
        # 缓存由于动态环境引起波动变化前的原始基准成本开销矩阵
        self._base_costs_data = np.copy(self.costs.data)
        self._base_true_costs_data = np.copy(self.true_costs.data)
        # 初始化与环境对应的动态事件引擎状态
        self.dynamic_event_engine.reset(
            num_envs=self.number_of_envs,
            num_jobs=self.number_of_jobs,
            num_machines=self.number_of_machines,
        )
        # 初始化阶段执行一次空的动态事件迭代计算下界，预热上下文
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
        # 复位并重置动态引擎的内部计数与种子分布
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
        # 环境步进结束后，依据已采样动作的后果评估动态事件的发动并在环境矩阵中生效它们
        self._apply_dynamic_events_after_transition(done_mask=done_mask, apply_sampling=True)
        # 将动态事件相关的额外上下文同步拼接保存到状态输出对象，以送入给 Hyper-Actor
        self._refresh_state_with_event_context()
        return self.state, reward, done

    def _apply_dynamic_events_after_transition(self, done_mask, apply_sampling: bool):
        # 每步过渡结束后实施所有环境扰动变体
        # 如果需要触发采集，从随机概率分布中抛出掷骰结果，反之只索取一个空白的安全状态
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

        # 针对【订单交期平移 Deadline Shift】类事件的环境数组变异处理
        if snapshot.deadline_delta.shape == self.deadlines.shape:
            self.deadlines = np.maximum(self.deadlines + snapshot.deadline_delta, 1e-8)
            self.true_deadlines = (
                self.deadlines * (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
                + self.pt_lower_bound
            )
            self.compute_tardiness_features()
            self.compute_flowtime_features()
            self.construct_op_features()

        # 针对【能源开销暴涨 Energy Spike】类事件的能耗张量扰动处理，用于推算动态成本 LB
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

        # 针对【机器故障 Breakdown】类事件阻断有效处理掩码图（将故障机器对应操作屏蔽掉）
        prev_dynamic_pair_mask = np.copy(self.dynamic_pair_mask)
        breakdown_mask = snapshot.breakdown_mask
        if breakdown_mask.shape == (self.number_of_envs, self.number_of_machines):
            self.dynamic_pair_mask = np.logical_or(
                self.dynamic_pair_mask, breakdown_mask[:, None, :]
            )
            # 防止全部机器断合规导致非法遮罩
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
        """收集引擎产生的多端全局动态上下文特征，组装拼接回供RL处理的前端特征变量 State"""
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

    def construct_op_features(self) -> None:
        """
        construct operation raw features
        """
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

        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def compute_tardiness_features(self) -> None:
        """针对优化目标中包含拖期或提早目标的要求，下发工件级别拖后情况的先验评估特征。"""
        # op_lateness_lb 目前用于评估 Tardiness 和 Earliness 等目标需求
        if (
            ObjectiveFn.TOTAL_TARDINESS in self.objectives
            or ObjectiveFn.TOTAL_EARLINESS in self.objectives
        ):
            # 通过估算的所属工件最终完成时间下界与给定交期 deadlines 的差值，求出各工件可能产生的 Lateness
            job_lateness = (
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.deadlines
            )
            # 通过 job_first_op_id 和 job_last_op_id 建立操作 op 到工件 job 的索引映射
            ops = np.arange(self.number_of_ops)[
                None, None, :
            ]  # 形状 (1, 1, 操作总数)
            job_first = self.job_first_op_id[
                ..., None
            ]  # 形状 (环境数量, 工件数量, 1)
            job_last = self.job_last_op_id[
                ..., None
            ]  # 形状 (环境数量, 工件数量, 1)
            job_mask = (ops >= job_first) & (
                ops <= job_last
            )  # 形状 (环境数量, 工件数量, 操作总数)
            op_job_id = job_mask.argmax(axis=1)  # 形状 (环境数量, 操作总数)
            # 把各个工件的 lateness 参数广播并赋发到其包含的所有操作特征上
            self.op_lateness_lb = job_lateness[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]  # 形状 (环境数量, 操作总数)

    def compute_flowtime_features(self) -> None:
        """针对优化平均流程时间目标，计算分配各工序对流程时间推算的特征表现。"""
        if ObjectiveFn.AVERAGE_FLOWTIME in self.objectives:
            # 如果该工件仍未被安排任何上机（启动时间为无穷大），则它的流程时长完全等于完工时间；
            # 其他情况下则需要减去工件的首次启动调度时刻求出跨度。
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
            # 同样构建操作 op 到 所属 job 的广播分发映射
            ops = np.arange(self.number_of_ops)[None, None, :]
            job_first = self.job_first_op_id[..., None]
            job_last = self.job_last_op_id[..., None]
            job_mask = (ops >= job_first) & (ops <= job_last)
            op_job_id = job_mask.argmax(axis=1)
            self.op_flowtime_lb = job_flowtime_lb[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]

    def set_initial_reward_quality(self) -> None:
        """
        初始化每个算例以及下阶段各目标函数的初步或基线评估（为计算奖励差值铺垫）。
        同时将估算出理想情况的下界（Lower bounds），以辅助算法在指标系中建立参照前沿。
        """
        # 如果处于简易版的直接奖励模式，起步均无任何分配调度，默认都定为0就行了
        if self.use_simple_reward:
            self.init_quality = np.zeros(
                (self.number_of_envs, self.nr_objectives), dtype=np.float64
            )
            # 仍旧评估理想完工的客观下界做为指标参考
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
            self.objective_lower_bounds = np.stack(objective_lower_bounds, axis=-1)
            return

        # 常规算法：基于完工时间下界的下限评估法（更为准确且便于收敛训练使用稠密信号）
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
                # 这个评估相当于在交期和最终完工之间做裁剪差，下限估计它的上界分布
                earliness_ub_estimates = np.maximum(
                    0,
                    self.deadlines
                    - self.op_ct_lb[
                        np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                    ],
                )
                quality = np.sum(earliness_ub_estimates, axis=1)
                # 总提早可能达到的极端最低下限可以为0（工单刚刚好压点全额交期完成）
                objective_lower_bounds.append(np.zeros(self.number_of_envs))
            elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                quality = -np.max(self.op_ct_lb, axis=1)
                objective_lower_bounds.append(-np.max(self.true_op_ct_lb, axis=1))
            elif obj == ObjectiveFn.COSTS:
                quality = np.copy(self.cost_lb)
                objective_lower_bounds.append(np.copy(self.true_cost_lb))
            else:
                raise ValueError(f"Unknown objective function: {obj}")
            qualities.append(quality)
        # 沿目标维度做特征拼接 (每个评估环境具备这几个维度的评估阵)
        self.init_quality = np.stack(qualities, axis=-1)
        self.objective_lower_bounds = np.stack(objective_lower_bounds, axis=-1)

    def compute_reward(self) -> np.ndarray:
        """
        根据环境设定来评估返回经过当前有效单步操作后各维度的差分奖励信息
        
        :return: 环境回馈数组张量 - 尺寸多目标为 [num_envs, num_objectives]
        """
        if self.use_simple_reward:
            return self._compute_simple_actual_value_reward()
        else:
            return self._compute_multi_objective_reward()

    def _compute_multi_objective_reward(self) -> np.ndarray:
        """
        计算多目标差分奖励（该环境默认的 PPO 实现途径，即连续增量法）。
        这里的推衍原则是：取每步调度新获得的目标质量质量（往往是下界的估定结果），与该目标执行之前的质量做减法比对，
        并乘上该目标的缩放系数（multiplier），所得正比增量作为奖励。

        :return: 多目标环境差分奖励序列矩阵 [num_envs, num_objectives]
        """
        rewards = []
        # 对于设定好的每个子目标
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
                # 评估提前完成情况的估计分布
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

            # 该目标奖励表现为相较于过去前一步在该评价品质上的减小改善（即指标下降 = 质量提升）
            reward = self.previous_obj_estimate[:, i] - current_quality
            reward *= self.multipliers[obj]
            rewards.append(reward)
            # 利用刚刚采集的数据覆盖以作为下步对冲推演的新鲜基准
            self.previous_obj_estimate[:, i] = current_quality

        # 将各实例及各个子目标计算所得拼合重组抛出结果
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

    def set_objective_weights(self, weights: np.ndarray) -> None:
        """
        Set objective weights

        :param weights: objective weights with shape [num_envs, num_objectives]
        """
        self.objective_weights = weights

    def _compute_current_qualities_with_multipliers(self) -> np.ndarray:
        """
        Compute current qualities for all objectives with multipliers applied.

        :return: array of shape [num_envs, num_objectives] with current qualities
        """
        qualities = []
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
            elif obj == ObjectiveFn.COSTS:
                current_quality = np.copy(self.cost_lb)
            else:
                current_quality = np.zeros(self.number_of_envs)

            # Apply multipliers
            current_quality = current_quality.astype(np.float64) * self.multipliers[obj]
            qualities.append(current_quality)

        return np.stack(qualities, axis=-1)  # [num_envs, num_objectives]
