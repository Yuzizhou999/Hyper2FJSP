import copy
import time

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


class MOTrainer(train.Trainer):
    """
    多目标强化学习调度训练的执行器类。
    继承自单目标/基石训练器 train.Trainer，增加了处理多目标优化的超网络架构与奖励折算。
    """
    def __init__(self, config):
        """
        初始化多目标训练器

        :param config: 参数配置对象，包含优化目标、硬件、以及相关的环境/网络架构参数
        """
        super().__init__(config, multi_objective=True)
        self.device = config.device

    def _set_objective_fn(self, objective_fns: list[str]):
        """
        将配置传入的目标字符串解析为 ObjectiveFn 枚举数组。
        同时保存目标函数的数量。
        """
        self.objective_fn = [
            ObjectiveFn(str(obj_fn).lower()) for obj_fn in objective_fns
        ]
        self.num_objectives = len(self.objective_fn)

    def _set_environments(self):
        """
        初始化训练与评估环境。
        支持从配置动态传递动态事件引擎的所有开启参数(如故障、插单、交期等)。
        """
        use_lb_features = getattr(self.config, "use_lb_features", True)
        
        # 构建给环境传入的动态事件参数配置字典
        dynamic_kwargs = {
            "dynamic_events_enabled": getattr(
                self.config, "dynamic_events_enabled", False
            ),
            "dynamic_event_prob": getattr(self.config, "dynamic_event_prob", 0.0),
            "dynamic_event_types": getattr(
                self.config,
                "dynamic_event_types",
                [
                    "job_arrival",
                    "machine_breakdown",
                    "deadline_shift",
                    "energy_spike",
                ],
            ),
            "dynamic_breakdown_duration": getattr(
                self.config, "dynamic_breakdown_duration", 2
            ),
            "dynamic_deadline_shift_scale": getattr(
                self.config, "dynamic_deadline_shift_scale", 0.1
            ),
            "dynamic_energy_duration": getattr(self.config, "dynamic_energy_duration", 2),
            "dynamic_energy_scale": getattr(self.config, "dynamic_energy_scale", 0.2),
            "dynamic_seed": getattr(self.config, "seed_train", None),
        }

        self.env = MOFJSPEnvForSameOpNums(
            self.n_j,
            self.n_m,
            objective_fns=self.objective_fn,
            data_source=self.data_source,
            use_simple_reward=getattr(self.config, "use_simple_reward", False),
            use_lb_features=use_lb_features,
            **dynamic_kwargs,
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
                **dynamic_kwargs,
            )
        elif self.data_source == "SD2":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
                **dynamic_kwargs,
            )
        elif self.data_source == "JSSP":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
                **dynamic_kwargs,
            )
        elif self.data_source == "FFSP":
            self.vali_env = MOFJSPEnvForSameOpNums(
                self.n_j,
                self.n_m,
                objective_fns=self.objective_fn,
                data_source=self.data_source,
                use_lb_features=use_lb_features,
                **dynamic_kwargs,
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
        """
        根据不同的设置项目组合并构造详细的模型名称，用于日志记录和存盘区分。
        编码了：目标种类、基础网络、超网络调节策略、事件开启情况、简单奖励等指示符。
        """
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

        # 检查是否关闭小下界（Layer Bound）启发式特征
        no_lb_suffix = "_NoLB" if not getattr(config, "use_lb_features", True) else ""
        
        # 将动态事件类型的开头几位缩写编入名称
        dynamic_suffix = ""
        if getattr(config, "dynamic_events_enabled", False):
            dynamic_suffix = "_dyn"
            event_types = getattr(config, "dynamic_event_types", [])
            if isinstance(event_types, list) and len(event_types) > 0:
                event_abbr = "".join([evt.split("_")[0][:3] for evt in event_types])
                dynamic_suffix += f"_{event_abbr}"

        # 加入超网络所使用的各项调节特性标志词
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
            if getattr(config, "hyper_event_adapter_enabled", False):
                hyper_mode_suffix += "_evt_hyper"

        # Create the model name with the objective suffix
        self.model_name = f"{self.data_name}{strToSuffix(config.model_suffix)}_{config.model_architecture.value}{hyper_mode_suffix}{dynamic_suffix}{'_no_trans' if config.use_gamma_beta is False else ''}{'_large' if config.hidden_dim_actor > 65 else ''}{obj_suffix}{single_critic_suffix}{simple_reward_suffix}{no_lb_suffix}_MO"

    def _set_initial_rewards(self):
        """
        回合奖励重置。
        我们只追踪这回合内智能体实际挣得的多目标奖励累积量。
        在强化学习更新中，初始基线（-init_quality）已作为环境信息处理，不在打印与统计的回合奖励里。
        """
        ep_rewards = np.zeros((self.num_envs, self.num_objectives))

        return ep_rewards

    def _initialize_episode(self, i_update):
        """
        每一环境交互周期（Episode）开始前重置初始状态、目标偏好向量并重新分配。
        
        :param i_update: 当前PPO进行的第 i 次模型更新迭代步
        :return: reset 后的新调度状态和计时信息
        """
        ep_st = time.time()
        # 每隔固定迭代步数对训练样本的数据规模/操作数等分布重新采样一次，增加多样性
        if i_update % self.reset_env_timestep == 0:
            dataset_job_length, dataset_op_pt = self.sample_training_instances()
            state = self.env.set_initial_data(
                dataset_job_length, dataset_op_pt, deadline_alpha=self.deadline_alpha
            )
        else:
            state = self.env.reset()

        # 生成新的且归一化为和为 1 的均匀采样偏好向量
        objective_weights = np.random.rand(self.num_envs, self.num_objectives)
        self.objective_weights = objective_weights / objective_weights.sum(
            axis=1, keepdims=True
        )
        self.objective_weights_tensor = (
            torch.from_numpy(self.objective_weights).float().to(self.device)
        )

        # 针对基于超网络的模型，本回合预分配一次生成权重以此节约算力
        if hasattr(self.ppo.policy, "assign_preferences"):
            self.ppo.policy.assign_preferences(self.objective_weights_tensor)
            self.ppo.policy_old.assign_preferences(self.objective_weights_tensor)

        return state, ep_st

    def _forward_pass(self, state):
        """
        进行模型前向传递出动作分布。
        自动辨别基础网络与超网络所需要的传参形式，同时支持传入环境事件残差。
        """
        # For hypernetwork models, use pre-generated parameters (preferences=None)
        # For other models, pass preferences normally
        prefs = (
            None
            if hasattr(self.ppo.policy, "assign_preferences")
            else self.objective_weights_tensor
        )

        policy_kwargs = {
            "fea_j": state.fea_j_tensor,  # [sz_b, N, 8]
            "op_mask": state.op_mask_tensor,  # [sz_b, N, N]
            "candidate": state.candidate_tensor,  # [sz_b, J]
            "fea_m": state.fea_m_tensor,  # [sz_b, M, 6]
            "mch_mask": state.mch_mask_tensor,  # [sz_b, M, M]
            "comp_idx": state.comp_idx_tensor,  # [sz_b, M, M, J]
            "dynamic_pair_mask": state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
            "fea_pairs": state.fea_pairs_tensor,  # [sz_b, J, M]
            "preferences": prefs,
        }
        if (
            hasattr(self.ppo.policy_old, "supports_event_context")
            and self.ppo.policy_old.supports_event_context
        ):
            policy_kwargs["event_context"] = state.event_context_tensor

        pi_envs, vals_envs = self.ppo.policy_old(
            **policy_kwargs
        )
        return pi_envs, vals_envs

    def _process_reward(self, reward):
        """
        处理从环境中获得的奖励矩阵。
        由于是多目标，环境返回的是目标个数维度的奖励向量。我们按照本次episode采样的随机权重(objective_weights)对这些奖励求加权和。
        
        :param reward: 多目标奖励阵列 [num_envs, num_objectives]
        :return: 转化为标量的回合综合奖励 [num_envs]
        """
        # 使用权重求和折算为单个标量以使用PPO标准更新
        reward = (reward * self.objective_weights).sum(axis=1)
        return reward

    def _call_update_ppo(self):
        """
        执行一轮PPO参数的更新。
        会将当前回合的偏好权重传入，使得网络除了从轨迹(memory)中学习状态外，还能结合当时的偏好正确对梯度定方向。
        """
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
        """
        在具有相同工序数结构的验证集上评估模型当前的帕累托表现性能（使用超体积HV作为主指标）
        
        :param objective_fn: 用于本次计算的目标函数枚举列表
        :return: 每张算例图算出的正态负超体积HV数组（方便以越小越优形式融合到训练曲线中）
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()
        
        # ====== 离散化评估偏好的构造 ======
        # 根据目标的数量生成对应维度的偏好测试采样点矩阵
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
                    f"尚未支持的偏好采样数: {self.num_preferences_validation}"
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

        # ====== 为包含超网络(HYPER_DANIEL)配置提前生成权重 ======
        if hasattr(self.ppo.policy, "assign_preferences"):
            # 将生成的权重按照环境Batch Size展开重复：[num_envs * num_prefs, pref_dim]
            repeated_preferences = eval_preferences.repeat(self.num_envs_validation, 1)
            self.ppo.policy.assign_preferences(repeated_preferences)

        # ====== 贪婪执行验证阶段前向传播 ======
        while True:
            with torch.no_grad():
                policy_kwargs = {
                    "fea_j": state.fea_j_tensor,  # [sz_b, N, 8]
                    "op_mask": state.op_mask_tensor,
                    "candidate": state.candidate_tensor,  # [sz_b, J]
                    "fea_m": state.fea_m_tensor,  # [sz_b, M, 6]
                    "mch_mask": state.mch_mask_tensor,  # [sz_b, M, M]
                    "comp_idx": state.comp_idx_tensor,  # [sz_b, M, M, J]
                    "dynamic_pair_mask": state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                    "fea_pairs": state.fea_pairs_tensor,
                    # 如果使用超网络结构则预生成的偏好缓存在网络内部，此处置空
                    "preferences": None
                    if hasattr(self.ppo.policy, "assign_preferences")
                    else eval_preferences.repeat(
                        state.fea_j_tensor.shape[0] // eval_preferences.shape[0], 1
                    ),
                }
                # 如果超网络包含事件适配器网络，也把状态推给它
                if (
                    hasattr(self.ppo.policy, "supports_event_context")
                    and self.ppo.policy.supports_event_context
                ):
                    policy_kwargs["event_context"] = state.event_context_tensor
                pi, _ = self.ppo.policy(**policy_kwargs)  # 返回各配对操作的概率分布 [sz_b, J, M]

            # 评估使用贪心选择，而不是按概率采样，力求稳定的评估结果
            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        
        # ====== 记录与计算各验证目标的超体积 (Hypervolume) ======
        eval_objectives = []
        reference_point_indices = []
        lb_indices = []  # 记录小下界中此评估目标代表的槽位索引

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
            # 将对应松紧度（deadline_alpha）匹配映射至在参考点坐标数组中的列号索引
            # 索引 3: alpha=0.5, 索引 4: alpha=0.9, 索引 5: alpha=1.0
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
            
        # 复制上述参考索引用于针对各个目标进行标准化
        normalization_values = copy.deepcopy(reference_point_indices)
        
        if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
            eval_objectives.append(-self.vali_env.current_makespan)
            reference_point_indices.append(-1)
            normalization_values.append(0)
            lb_indices.append(
                self.vali_env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN)
            )
            
        # [num_envs_validation * num_preferences_validation, num_objectives]
        eval_objectives = np.stack(
            eval_objectives, axis=1
        )
        
        # 降维分离出各验证图自身所产生的不同权重点值，以便后续每张图各自求HV
        # [num_envs_validation, num_preferences_validation, num_objectives]
        eval_objectives = eval_objectives.reshape(
            self.num_envs_validation, self.num_preferences_validation, -1
        )
        normalized_hypervolumes = []
        for i in range(eval_objectives.shape[0]):
            # 调整下边界以匹配 eval_objectives 的收集循序列表
            all_lower_bounds = self.vali_env.objective_lower_bounds[
                i * self.num_preferences_validation
            ]
            lower_bounds = all_lower_bounds[lb_indices]
            
            # 使用基于下边界缩放过的各个偏好的目标空间和它的参考点算正态HV值。
            # 首先剔除产生集内部是被支配的劣质点(find_pareto_efficient_solutions)，然后再缩放求HV。
            normalized_hypervolume = compute_hypervolume(
                find_pareto_efficient_solutions(eval_objectives[i]) - lower_bounds,
                self.vali_reference_points[i][reference_point_indices] - lower_bounds,
            ) / np.prod(
                self.vali_reference_points[i][normalization_values] - lower_bounds
            )
            
            # 收集每个验证环境的负HV，因在机器学习里 loss 通常是越小越好
            normalized_hypervolumes.append(-normalized_hypervolume)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        return normalized_hypervolumes

    def validate_envs_with_various_op_nums(self, objective_fn):
        """
        验证集逻辑同上，区别在于专门用于不同工序数目动态大小的测试集环境。
        主要用于衡量零样本泛化能力 (Zero-shot generalizability)。
        """
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

        # ====== 为包含超网络(HYPER_DANIEL)配置提前生成权重，节约内存 ======
        if hasattr(self.ppo.policy, "assign_preferences"):
            # 在各类工序的推断中，由于环境中可能出现工件完成从而使Batch Size变动，
            # 传递整个重叠偏好数组容易对齐越界(OOM)。这里的做法是调用并缓存唯一的一组[num_prefs, pref_dim]，
            # 然后使用基于取余算子的 preference_indices 选取机制即可。
            self.ppo.policy.assign_preferences(eval_preferences)
            use_preference_indices = True
        else:
            use_preference_indices = False

        while True:
            with torch.no_grad():
                # 计算当前处于 active 状态序列的环境索引掩码，用来提取输入
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)

                if use_preference_indices:
                    # 使用内部缓存，将环境索引通过模运算映射回它当前测试的唯一前沿点
                    active_indices = torch.where(batch_idx)[0]
                    # 由于环境是重复打包成的 [env0_pref0, env0_pref1, ..., env1_pref0]
                    # 取模即可找到对应的 preference索引
                    pref_indices = active_indices % self.num_preferences_validation

                    policy_kwargs = {
                        "fea_j": state.fea_j_tensor[batch_idx],
                        "op_mask": state.op_mask_tensor[batch_idx],
                        "candidate": state.candidate_tensor[batch_idx],
                        "fea_m": state.fea_m_tensor[batch_idx],
                        "mch_mask": state.mch_mask_tensor[batch_idx],
                        "comp_idx": state.comp_idx_tensor[batch_idx],
                        "dynamic_pair_mask": state.dynamic_pair_mask_tensor[batch_idx],
                        "fea_pairs": state.fea_pairs_tensor[batch_idx],
                        "preferences": None,  # 直接使用内部分配机制
                        "preference_indices": pref_indices,  # 映射索引
                    }
                    if (
                        hasattr(self.ppo.policy, "supports_event_context")
                        and self.ppo.policy.supports_event_context
                    ):
                        policy_kwargs["event_context"] = state.event_context_tensor[
                            batch_idx
                        ]
                    pi, _ = self.ppo.policy(**policy_kwargs)
                else:
                    # 非超网络模型: 退回原本直接传入整个Preferences矩阵（可能会随batch缩小遇到不对齐尺寸麻烦）
                    policy_kwargs = {
                        "fea_j": state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                        "op_mask": state.op_mask_tensor[batch_idx],
                        "candidate": state.candidate_tensor[batch_idx],  # [sz_b, J]
                        "fea_m": state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                        "mch_mask": state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                        "comp_idx": state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                        "dynamic_pair_mask": state.dynamic_pair_mask_tensor[
                            batch_idx
                        ],  # [sz_b, J, M]
                        "fea_pairs": state.fea_pairs_tensor[batch_idx],
                        "preferences": eval_preferences.repeat(
                            state.fea_j_tensor[batch_idx].shape[0]
                            // eval_preferences.shape[0],
                            1,
                        ),
                    }
                    if (
                        hasattr(self.ppo.policy, "supports_event_context")
                        and self.ppo.policy.supports_event_context
                    ):
                        policy_kwargs["event_context"] = state.event_context_tensor[
                            batch_idx
                        ]
                    pi, _ = self.ppo.policy(**policy_kwargs)  # [sz_b, J, M]
            
            # 使用动作概率分布作贪婪选取推断
            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        
        # ====== 提取多目标并计算最终正规化体积 ======
        eval_objectives = []
        reference_point_indices = []
        lb_indices = []  # 记录小下界中此评估目标代表的槽位索引

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
            # 同样按照严格程度关联参考点坐标的列
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
        # 将结构变换回每个实例各自一组目标空间点的形式
        eval_objectives = eval_objectives.reshape(
            self.num_envs_validation, self.num_preferences_validation, -1
        )
        normalized_hypervolumes = []
        for i in range(eval_objectives.shape[0]):
            # 重新排列对齐下边界的索引顺序
            all_lower_bounds = self.vali_env.objective_lower_bounds[
                i * self.num_preferences_validation
            ]
            lower_bounds = all_lower_bounds[lb_indices]
            
            # 使用各自实例参考点结合解群集求HV（先去掉受支配的解）
            normalized_hypervolume = compute_hypervolume(
                find_pareto_efficient_solutions(eval_objectives[i]) - lower_bounds,
                self.vali_reference_points[i][reference_point_indices] - lower_bounds,
            ) / np.prod(
                self.vali_reference_points[i][normalization_values] - lower_bounds
            )
            # 取负以适应Loss最小化评价趋势
            normalized_hypervolumes.append(-normalized_hypervolume)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        return normalized_hypervolumes


def main():
    """主程序入口，构建多目标训练器实例并根据配置启动训练。"""
    trainer = MOTrainer(train.configurations)
    trainer.train()


if __name__ == "__main__":
    main()
