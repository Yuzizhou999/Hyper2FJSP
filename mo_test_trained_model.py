import copy
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from common_utils import (
    das_dennis,
    find_pareto_efficient_solutions,
    greedy_select_action,
    is_pareto_efficient,
    sample_action,
    setup_seed,
)
from data_utils import pack_data_from_config
from enums import ObjectiveFn
from hypervolume_utils import compute_hypervolume
from mo_fjsp_env_same_op_nums import MOFJSPEnvForSameOpNums
from model.PPO import PPO_initialize
from params import configs

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


device = torch.device(configs.device)

ppo = PPO_initialize(multi_objective=True)
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def resolve_test_seeds(config):
    explicit_seeds = getattr(config, "test_seed_list", None)
    if explicit_seeds:
        return [int(seed) for seed in explicit_seeds]

    base_seed = int(getattr(config, "seed_test", 0))
    num_scenarios = int(max(1, getattr(config, "num_test_scenarios", 1)))
    seed_stride = int(max(1, getattr(config, "test_seed_stride", 1)))
    return [base_seed + i * seed_stride for i in range(num_scenarios)]


def save_pareto_sets(pareto_sets, data_source, data_name, model_name, strategy):
    """
    将生成的帕累托前沿集合（Pareto Sets）保存到本地磁盘。

    :param pareto_sets: 每个测试用例计算得出的帕累托解集组成的列表
    :param data_source: 数据分布来源 (例如 'SD1', 'SD2', 'JSSP')
    :param data_name: 使用的数据集的具体名字 (例 '10x5')
    :param model_name: 被测试的模型名字
    :param strategy: 所用的推理策略（'greedy'，即贪婪模式，或其他采样策略）
    """
    save_dir = f"./pareto_sets/{data_source}/{data_name}/{model_name}_{strategy}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, pareto_set in enumerate(pareto_sets):
        save_path = f"{save_dir}/instance_{i}.npy"
        # Extract data from masked array if needed
        pareto_set_data = pareto_set.data if hasattr(pareto_set, "data") else pareto_set
        np.save(save_path, pareto_set_data)

    print(f"Saved {len(pareto_sets)} Pareto sets to {save_dir}")


def save_preference_response(
    preferences,
    objectives,
    pareto_mask,
    objective_names,
    data_source,
    data_name,
    model_name,
    strategy,
    instance_index,
):
    """
    保存一个测试实例下，偏好向量与其对应的客观评价值的响应关系。
    这用于后续分析超网络对不同倾向(权重)设定是否具备有效并准确的方向响应能力。

    :param preferences: 形状为 [num_points, pref_dim] 的偏好向量记录
    :param objectives: 基于偏好决策实际产生的解目标阵列 [num_points, num_objectives]
    :param pareto_mask: 布尔掩码标注哪些结果解落在了帕累托非支配前沿上
    :param objective_names: 目标名称（按列排列一致）
    """
    save_dir = f"./preference_responses/{data_source}/{data_name}/{model_name}_{strategy}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/instance_{instance_index}.npz"
    np.savez(
        save_path,
        preferences=np.asarray(preferences),
        objectives=np.asarray(objectives),
        pareto_mask=np.asarray(pareto_mask, dtype=bool),
        objective_names=np.asarray(objective_names),
    )


def collect_eval_objectives(env, objective_fn, cost_reference_index=None):
    """
    工具方法：汇总验证环境测试跑完后，所获得的各个目标评级的真实值并映射相关归一化引用的索引。
    
    :return: 
      - eval_objectives: 收集完成的目标多轴值矩阵
      - reference_point_indices: 具体目标在参考基准点数组中的列排布索引
      - lb_indices: Lower-bound特征索引
      - normalization_values: 映射用于后续参考点正态化的基准索引
      - objective_names: 各被采纳目标的字符串英文缩写名称组成的列表
    """
    eval_objectives = []
    reference_point_indices = []
    lb_indices = []
    objective_names = []

    if ObjectiveFn.MAKESPAN in objective_fn:
        eval_objectives.append(env.current_makespan)
        reference_point_indices.append(0)
        lb_indices.append(env.objectives.index(ObjectiveFn.MAKESPAN))
        objective_names.append(ObjectiveFn.MAKESPAN.value)
    if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn:
        eval_objectives.append(env.compute_avg_flowtime())
        reference_point_indices.append(1)
        lb_indices.append(env.objectives.index(ObjectiveFn.AVERAGE_FLOWTIME))
        objective_names.append(ObjectiveFn.AVERAGE_FLOWTIME.value)
    if ObjectiveFn.TOTAL_TARDINESS in objective_fn:
        eval_objectives.append(env.compute_total_tardiness())
        reference_point_indices.append(2)
        lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_TARDINESS))
        objective_names.append(ObjectiveFn.TOTAL_TARDINESS.value)
    if ObjectiveFn.TOTAL_EARLINESS in objective_fn:
        eval_objectives.append(env.compute_total_earliness())
        if configs.deadline_alpha <= 0.51:
            reference_point_indices.append(3)
        elif configs.deadline_alpha <= 0.91:
            reference_point_indices.append(4)
        else:
            reference_point_indices.append(5)
        lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_EARLINESS))
        objective_names.append(ObjectiveFn.TOTAL_EARLINESS.value)
    if ObjectiveFn.COSTS in objective_fn:
        eval_objectives.append(env.compute_costs())
        if cost_reference_index is not None:
            reference_point_indices.append(cost_reference_index)
        lb_indices.append(env.objectives.index(ObjectiveFn.COSTS))
        objective_names.append(ObjectiveFn.COSTS.value)
    normalization_values = copy.deepcopy(reference_point_indices)
    if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
        eval_objectives.append(-env.current_makespan)
        reference_point_indices.append(-1)
        normalization_values.append(0)
        lb_indices.append(env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN))
        objective_names.append(ObjectiveFn.NEGATIVE_MAKESPAN.value)

    eval_objectives = np.stack(eval_objectives, axis=1)
    return (
        eval_objectives,
        reference_point_indices,
        lb_indices,
        normalization_values,
        objective_names,
    )


def test_greedy_strategy(
    data_set,
    model_path,
    seed,
    objective_fn: list[ObjectiveFn],
    reference_points,
    model_name=None,
    data_source=None,
    data_name=None,
    num_preferences=101,
    strategy_tag="greedy",
):
    """
    使用贪婪策略（Greedy Strategy）在提供的测试数据集上测试并评估训练好的超网络强化模型。反向使用推断所得的目标解求取前沿。

    :param data_set: 测试用的数据集 (包含工件与工序时长等矩阵)
    :param model_path: 已保存的 PyTorch 模型文件路径
    :param seed: 随机种子，用于复现固定环境
    :param objective_fn: 对应环境与评价要使用的目标集合
    :param reference_points: HV或性能计算里用于比较归一化上限的参考点信息
    :param model_name: 模型日志名称
    :param data_source: 验证数据分布名 (比如 JSSP, SD1, SD2)
    :param data_name: 数据子集名 (比如 '15x15')
    :param num_preferences: 偏好的划分密度/离散个数（默认为两个目标的 101 等分）
    :return: 包含各种评估指标均值方差的性能字典 performance_metrics 
    """

    test_result_list = []

    setup_seed(seed)
    # 重载 PPO 里的Actor/Critic参数用于前向
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
    ppo.policy.eval()

    # Detect use_lb_features from model name
    use_lb_features = "_NoLB" not in model_name if model_name else True

    # 实例维度还原
    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    
    # 构建兼容多目标推断与各类随机事件开启的 FJSP 环境
    env = MOFJSPEnvForSameOpNums(
        n_j=n_j,
        n_m=n_m,
        objective_fns=objective_fn,
        data_source=data_source,
        use_lb_features=use_lb_features,
        dynamic_events_enabled=getattr(configs, "dynamic_events_enabled", False),
        dynamic_event_prob=getattr(configs, "dynamic_event_prob", 0.0),
        dynamic_event_types=getattr(
            configs,
            "dynamic_event_types",
            ["job_arrival", "machine_breakdown", "deadline_shift", "energy_spike"],
        ),
        dynamic_breakdown_duration=getattr(configs, "dynamic_breakdown_duration", 2),
        dynamic_deadline_shift_scale=getattr(
            configs, "dynamic_deadline_shift_scale", 0.1
        ),
        dynamic_energy_duration=getattr(configs, "dynamic_energy_duration", 2),
        dynamic_energy_scale=getattr(configs, "dynamic_energy_scale", 0.2),
        dynamic_seed=seed,
    )

    performance_metrics = defaultdict(list)

    # =============== 初始化用于多目标的评价散点偏好 ===============
    if len(objective_fn) == 2:
        # 两目标直接均匀切分
        preferences = np.linspace(0, 1, num_preferences)
        preferences = np.array([preferences, 1 - preferences]).T
        preferences = torch.from_numpy(preferences).float().to(device)
    elif len(objective_fn) == 3:
        # 三目标使用 das_dennis 对三角流形内采样
        if num_preferences == 105:
            preferences = das_dennis(13, 3)  # 105
        elif num_preferences == 496:
            preferences = das_dennis(30, 3)
        elif num_preferences == 1035:
            preferences = das_dennis(44, 3)
        elif num_preferences == 45:
            preferences = das_dennis(8, 3)
        elif num_preferences == 10:
            preferences = das_dennis(3, 3)
        elif num_preferences == 10011:
            preferences = das_dennis(140, 3)
        else:
            num_preferences = 105
            preferences = das_dennis(13, 3)

        preferences = torch.from_numpy(preferences).float().to(device)
    elif len(objective_fn) == 4:
        # 四目标类似采样
        if num_preferences == 84:
            preferences = das_dennis(6, 4)  # 84
        elif num_preferences == 120:
            preferences = das_dennis(7, 4)
        elif num_preferences == 56:
            preferences = das_dennis(5, 4)  # 56
        elif num_preferences == 35:
            preferences = das_dennis(4, 4)
        elif num_preferences == 20:
            preferences = das_dennis(3, 4)
        elif num_preferences == 15:
            preferences = das_dennis(2, 4)
        else:
            num_preferences = 120
            preferences = das_dennis(7, 4)

        preferences = torch.from_numpy(preferences).float().to(device)
    else:
        raise NotImplementedError("目前不支持大于四个目标的偏好矩阵采样模式。")
        
    preferences_np = preferences.detach().cpu().numpy()
    original_data_set_size = len(data_set[0])
    
    # 数据增强：使用复制的方式，让每一个 Batch Size (101个解) 处理的都是同一个拓扑不同的偏好
    data_set = (
        [i for i in data_set[0] for _ in range(num_preferences)],
        [i for i in data_set[1] for _ in range(num_preferences)],
    )

    pareto_sets = []

    # ============== 对于搭载超网络的架构预分配权重 ==============
    if hasattr(ppo.policy, "assign_preferences"):
        ppo.policy.assign_preferences(preferences)
        prefs_to_pass = None
    else:
        prefs_to_pass = preferences

    # ============== 进行针对各个样本的具体贪婪推断遍历 ==============
    for i in tqdm(
        range(original_data_set_size), file=sys.stdout, desc="progress", colour="blue"
    ):
        state = env.set_initial_data(
            data_set[0][i * num_preferences : (i + 1) * num_preferences],
            data_set[1][i * num_preferences : (i + 1) * num_preferences],
            deadline_alpha=configs.deadline_alpha,
        )
        t1 = time.time()
        while True:
            # PPO Actor 的评分提取
            with torch.no_grad():
                policy_kwargs = {
                    "fea_j": state.fea_j_tensor,  # [num_preferences, N, 8]
                    "op_mask": state.op_mask_tensor,  # [num_preferences, N, N]
                    "candidate": state.candidate_tensor,  # [num_preferences, J]
                    "fea_m": state.fea_m_tensor,  # [num_preferences, M, 6]
                    "mch_mask": state.mch_mask_tensor,  # [num_preferences, M, M]
                    "comp_idx": state.comp_idx_tensor,  # [num_preferences, M, M, J]
                    "dynamic_pair_mask": state.dynamic_pair_mask_tensor,
                    "fea_pairs": state.fea_pairs_tensor,
                    "preferences": prefs_to_pass,  # none 给超网络, preferences 给基模
                }
                
                # 若开启事件环境响应网络
                if (
                    hasattr(ppo.policy, "supports_event_context")
                    and ppo.policy.supports_event_context
                ):
                    policy_kwargs["event_context"] = state.event_context_tensor
                    
                # [num_preferences, J, M] : 各组合偏好对应的调度策略概率分布
                pi, _ = ppo.policy(**policy_kwargs)  

            # 根据贪心策略选择分布中概率最大的合法动作为最终采取的动作
            action = greedy_select_action(pi)
            state, reward, done = env.step(actions=action.cpu().numpy())
            if done.all():
                break
        t2 = time.time()
        (
            eval_objectives,
            reference_point_indices,
            lb_indices,
            normalization_values,
            objective_names,
        ) = collect_eval_objectives(
            env, objective_fn, cost_reference_index=len(reference_points[0]) - 1
        )
        per_preference_objectives = eval_objectives.reshape(num_preferences, -1)
        pareto_mask = is_pareto_efficient(per_preference_objectives)
        eval_objectives = per_preference_objectives.reshape(1, num_preferences, -1)
        normalized_hypervolumes = []
        for j in range(eval_objectives.shape[0]):
            print(
                f"Num unique solutions: {len(np.unique(eval_objectives[j], axis=0))}; Number in pareto_set: {len(find_pareto_efficient_solutions(eval_objectives[j]))}"
            )
            pareto_set = find_pareto_efficient_solutions(eval_objectives[j])
            pareto_sets.append(pareto_set)  # Store the pareto set
            # Reorder lower bounds to match the order of objectives using lb_indices
            all_lower_bounds = env.objective_lower_bounds[0]
            lower_bounds = all_lower_bounds[lb_indices]
            normalized_hypervolume = compute_hypervolume(
                pareto_set - lower_bounds,
                reference_points[i][reference_point_indices] - lower_bounds + 1e-8,
            ) / np.prod(reference_points[i][normalization_values] - lower_bounds + 1e-8)
            normalized_hypervolumes.append(normalized_hypervolume)
            num_solutions_in_pareto_set = len(pareto_set)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        test_result_list.append(
            [normalized_hypervolume, num_solutions_in_pareto_set, t2 - t1]
        )
        if model_name and data_source and data_name:
            save_preference_response(
                preferences_np,
                per_preference_objectives,
                pareto_mask,
                objective_names,
                data_source,
                data_name,
                model_name,
                strategy_tag,
                i,
            )

    # Save the Pareto sets if model_name is provided
    if model_name and data_source and data_name:
        save_pareto_sets(pareto_sets, data_source, data_name, model_name, strategy_tag)

    return np.array(test_result_list), performance_metrics


def test_sampling_strategy(
    data_set,
    model_path,
    sample_times,
    seed,
    objective_fn: list[ObjectiveFn],
    reference_points,
    model_name=None,
    data_source=None,
    data_name=None,
    num_preferences=101,
    num_sampling_cycles=1,
    strategy_tag="sampling",
):
    test_result_list = []
    all_cycles_pareto_sets = []  # Store pareto sets for each cycle
    all_cycles_performance_metrics = []  # Store performance metrics for each cycle

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
    ppo.policy.eval()

    # Detect use_lb_features from model name
    use_lb_features = "_NoLB" not in model_name if model_name else True

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    env = MOFJSPEnvForSameOpNums(
        n_j=n_j,
        n_m=n_m,
        objective_fns=objective_fn,
        data_source=data_source,
        use_lb_features=use_lb_features,
        dynamic_events_enabled=getattr(configs, "dynamic_events_enabled", False),
        dynamic_event_prob=getattr(configs, "dynamic_event_prob", 0.0),
        dynamic_event_types=getattr(
            configs,
            "dynamic_event_types",
            ["job_arrival", "machine_breakdown", "deadline_shift", "energy_spike"],
        ),
        dynamic_breakdown_duration=getattr(configs, "dynamic_breakdown_duration", 2),
        dynamic_deadline_shift_scale=getattr(
            configs, "dynamic_deadline_shift_scale", 0.1
        ),
        dynamic_energy_duration=getattr(configs, "dynamic_energy_duration", 2),
        dynamic_energy_scale=getattr(configs, "dynamic_energy_scale", 0.2),
        dynamic_seed=seed,
    )

    performance_metrics = defaultdict(list)

    if len(objective_fn) == 2:
        preferences = np.linspace(0, 1, num_preferences)
        preferences = np.array([preferences, 1 - preferences]).T
        preferences = preferences.repeat(sample_times, axis=0)
        preferences = torch.from_numpy(preferences).float().to(device)
    elif len(objective_fn) == 3:
        if num_preferences == 105:
            preferences = das_dennis(13, 3)  # 105
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 496:
            preferences = das_dennis(30, 3)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 1035:
            preferences = das_dennis(44, 3)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 45:
            preferences = das_dennis(8, 3)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 10:
            preferences = das_dennis(3, 3)
            preferences = preferences.repeat(sample_times, axis=0)
        else:
            num_preferences = 105
            preferences = das_dennis(13, 3)
            preferences = preferences.repeat(sample_times, axis=0)

        preferences = torch.from_numpy(preferences).float().to(device)
    elif len(objective_fn) == 4:
        if num_preferences == 84:
            preferences = das_dennis(6, 4)  # 84
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 120:
            preferences = das_dennis(7, 4)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 56:
            preferences = das_dennis(5, 4)  # 56
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 35:
            preferences = das_dennis(4, 4)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 20:
            preferences = das_dennis(3, 4)
            preferences = preferences.repeat(sample_times, axis=0)
        elif num_preferences == 15:
            preferences = das_dennis(2, 4)
            preferences = preferences.repeat(sample_times, axis=0)
        else:
            num_preferences = 120
            preferences = das_dennis(7, 4)
            preferences = preferences.repeat(sample_times, axis=0)

        preferences = torch.from_numpy(preferences).float().to(device)
    else:
        raise NotImplementedError("Unsupported number of objectives.")
    preferences_np = preferences.detach().cpu().numpy()
    original_data_set_size = len(data_set[0])
    data_set = (
        [i for i in data_set[0] for _ in range(num_preferences * sample_times)],
        [i for i in data_set[1] for _ in range(num_preferences * sample_times)],
    )

    # 对于搭载 HYPER_DANIEL 超网络的架构：仅对所有唯一的偏好点生成一次参数即可
    # 注意：preferences 张量会根据采样次数 repeat 多次，但我们只抽取独一无二的值
    # preferences.repeat(sample_times, axis=0) 会创建: [p0,p0,...,p0, p1,p1,...,p1, ...]
    # 所以我们需要按 sample_times 的步长提取出唯一的偏好值
    if hasattr(ppo.policy, "assign_preferences"):
        # 沿 0 维以 sample_times 步长提取，获取所有独立的偏好值
        unique_preferences = preferences[
            ::sample_times
        ]  # 抽取索引 0, sample_times, 2*sample_times, ...
        ppo.policy.assign_preferences(unique_preferences)
        use_preference_indices = True
    else:
        use_preference_indices = False

    # 在处理下一个算例之前，完成属于该算例所有周期的采样过程
    for i in tqdm(
        range(original_data_set_size),
        file=sys.stdout,
        desc="Instance progress",
        colour="blue",
    ):
        # 累积当前算例在所有采样周期的求解结果
        accumulated_solutions = []  # 存放所有周期的所有解

        # 对当前算例执行多周期的多次采样寻优
        for cycle in range(num_sampling_cycles):
            if num_sampling_cycles > 1:
                print(
                    f"\n--- Instance {i + 1}/{original_data_set_size}, Cycle {cycle + 1}/{num_sampling_cycles} ---"
                )

            state = env.set_initial_data(
                data_set[0][
                    i * num_preferences * sample_times : (i + 1)
                    * num_preferences
                    * sample_times
                ],
                data_set[1][
                    i * num_preferences * sample_times : (i + 1)
                    * num_preferences
                    * sample_times
                ],
                deadline_alpha=configs.deadline_alpha,
            )
            t1 = time.time()
            while True:
                with torch.no_grad():
                    if use_preference_indices:
                        # Map each environment to its unique preference index
                        # Environments ordered (after repeat): [pref0,pref0,...,pref0, pref1,pref1,...,pref1, ...]
                        # So indices are: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...]
                        num_envs = state.fea_j_tensor.shape[0]
                        pref_indices = (
                            torch.arange(num_envs, device=device) // sample_times
                        )

                        policy_kwargs = {
                            "fea_j": state.fea_j_tensor,
                            "op_mask": state.op_mask_tensor,
                            "candidate": state.candidate_tensor,
                            "fea_m": state.fea_m_tensor,
                            "mch_mask": state.mch_mask_tensor,
                            "comp_idx": state.comp_idx_tensor,
                            "dynamic_pair_mask": state.dynamic_pair_mask_tensor,
                            "fea_pairs": state.fea_pairs_tensor,
                            "preferences": None,  # Already assigned
                            "preference_indices": pref_indices,
                        }
                        if (
                            hasattr(ppo.policy, "supports_event_context")
                            and ppo.policy.supports_event_context
                        ):
                            policy_kwargs["event_context"] = state.event_context_tensor
                        pi, _ = ppo.policy(**policy_kwargs)
                    else:
                        policy_kwargs = {
                            "fea_j": state.fea_j_tensor,  # [num_preferences * sample_times, N, 8]
                            "op_mask": state.op_mask_tensor,
                            "candidate": state.candidate_tensor,
                            "fea_m": state.fea_m_tensor,
                            "mch_mask": state.mch_mask_tensor,
                            "comp_idx": state.comp_idx_tensor,
                            "dynamic_pair_mask": state.dynamic_pair_mask_tensor,
                            "fea_pairs": state.fea_pairs_tensor,
                            "preferences": preferences,
                        }
                        if (
                            hasattr(ppo.policy, "supports_event_context")
                            and ppo.policy.supports_event_context
                        ):
                            policy_kwargs["event_context"] = state.event_context_tensor
                        pi, _ = ppo.policy(**policy_kwargs)

                action, _ = sample_action(pi)
                state, reward, done = env.step(actions=action.cpu().numpy())
                if done.all():
                    break
            t2 = time.time()
            (
                eval_objectives,
                reference_point_indices,
                lb_indices,
                normalization_values,
                objective_names,
            ) = collect_eval_objectives(
                env, objective_fn, cost_reference_index=len(reference_points[0]) - 1
            )

            eval_objectives = eval_objectives.reshape(num_preferences * sample_times, -1)

            # 把本周期的解添加到累积集合中
            accumulated_solutions.append(eval_objectives)

            # 沿纵向拼接合并迄今为止获得的所有解
            all_solutions_so_far = np.vstack(
                accumulated_solutions
            )  # 形状: [目前为止的总解数量, 目标维度]
            all_preferences_so_far = np.tile(preferences_np, (cycle + 1, 1))

            print(
                f"Cycle {cycle + 1}: Added {len(eval_objectives)} solutions. "
                f"Total accumulated: {len(all_solutions_so_far)} solutions. "
                f"Unique: {len(np.unique(all_solutions_so_far, axis=0))}"
            )

            # 根据所有累积起来的解计算最终的帕累托前沿集合
            pareto_set = find_pareto_efficient_solutions(all_solutions_so_far)
            pareto_mask = is_pareto_efficient(all_solutions_so_far)
            print(f"Accumulated Pareto set size: {len(pareto_set)}")

            # 根据累积的帕累托前沿求解超体积
            # 重新排序各维度的下界基准线以匹配验证集的观测指标顺序
            all_lower_bounds = env.objective_lower_bounds[0]
            lower_bounds = all_lower_bounds[lb_indices]
            normalized_hypervolume = compute_hypervolume(
                pareto_set - lower_bounds,
                reference_points[i][reference_point_indices] - lower_bounds + 1e-8,
            ) / np.prod(reference_points[i][normalization_values] - lower_bounds + 1e-8)
            num_solutions_in_pareto_set = len(pareto_set)

            print(f"Accumulated Normalized Hypervolume: {-normalized_hypervolume:.6f}")

            test_result_list.append(
                [normalized_hypervolume, num_solutions_in_pareto_set, t2 - t1]
            )

            # 保存当前算例于本周期的结果数据
            all_cycles_pareto_sets.append(pareto_set)
            all_cycles_performance_metrics.append(performance_metrics)

            # 如果存在文件指向信息，则保存此周期针对该算例累积出来的帕累托集合
            if model_name and data_source and data_name:
                # 若使用了 masked_array 取决于类型可萃取其中 data
                pareto_set_data = (
                    pareto_set.data if hasattr(pareto_set, "data") else pareto_set
                )

                if num_sampling_cycles > 1:
                    # 在保存路径中插入当前周期序号
                    save_dir = (
                        f"./pareto_sets/{data_source}/{data_name}/"
                        f"{model_name}_{strategy_tag}_cycle{cycle + 1}"
                    )
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/instance_{i}.npy"
                    np.save(save_path, pareto_set_data)
                else:
                    # 仅具有一轮采样循环的原有保存处理逻辑
                    save_dir = (
                        f"./pareto_sets/{data_source}/{data_name}/{model_name}_{strategy_tag}"
                    )
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/instance_{i}.npy"
                    np.save(save_path, pareto_set_data)

                save_preference_response(
                    all_preferences_so_far,
                    all_solutions_so_far,
                    pareto_mask,
                    objective_names,
                    data_source,
                    data_name,
                    model_name,
                    (
                        f"{strategy_tag}_cycle{cycle + 1}"
                        if num_sampling_cycles > 1
                        else strategy_tag
                    ),
                    i,
                )

    return np.array(test_result_list), performance_metrics


def main(config, flag_sample):
    """
    Load trained models, run tests, and save summarized outputs.
    :param flag_sample: True for sampling policy, False for greedy policy.
    """
    setup_seed(config.seed_test)
    objective_fn = [ObjectiveFn(str(obj_fn).lower()) for obj_fn in config.objective_fn]
    if not os.path.exists("./test_results"):
        os.makedirs("./test_results")

    test_model = []
    for model_name in config.test_model:
        test_model.append(
            (f"./trained_network/{config.model_source}/{model_name}.pth", model_name)
        )

    test_data, test_reference_points = pack_data_from_config(
        config.data_source, config.test_data, load_reference_points=True
    )

    if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
        updated_reference_points = []
        for reference_points, reference_name in test_reference_points:
            reference_points = np.array(reference_points)
            reference_points = np.concatenate(
                (reference_points, np.zeros((reference_points.shape[0], 1))),
                axis=1,
            )
            updated_reference_points.append((reference_points, reference_name))
        test_reference_points = updated_reference_points

    test_seeds = resolve_test_seeds(config)
    print(f"Test scenario seeds: {test_seeds}")

    model_prefix = "DANIELS" if flag_sample else "DANIELG"

    for data_idx, (data_payload, data_name) in enumerate(test_data):
        reference_points, reference_name = test_reference_points[data_idx]
        if data_name != reference_name:
            print(
                f"Warning: data/reference mismatch ({data_name} vs {reference_name}); "
                "using paired order from pack_data_from_config."
            )

        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data_name}")
        print(f"test mode: {model_prefix}")
        save_direc = f"./test_results/{config.data_source}/{data_name}"
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model_path, model_name in test_model:
            save_path = save_direc + f"/Result_{model_prefix}+{model_name}_{data_name}.npy"
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model_name}")
                print(f"data name: ./data/{config.data_source}/{data_name}")

                scenario_results = []
                aggregated_metrics = defaultdict(list)
                for seed in test_seeds:
                    if not flag_sample:
                        strategy_tag = (
                            "greedy" if len(test_seeds) == 1 else f"greedy_seed{seed}"
                        )
                        result, performance_metrics = test_greedy_strategy(
                            data_payload,
                            model_path,
                            seed,
                            objective_fn=objective_fn,
                            reference_points=np.array(reference_points),
                            model_name=model_name,
                            data_source=config.data_source,
                            data_name=data_name,
                            num_preferences=config.num_preferences_test,
                            strategy_tag=strategy_tag,
                        )
                    else:
                        strategy_tag = (
                            "sampling" if len(test_seeds) == 1 else f"sampling_seed{seed}"
                        )
                        result, performance_metrics = test_sampling_strategy(
                            data_payload,
                            model_path,
                            config.sample_times,
                            seed,
                            objective_fn=objective_fn,
                            reference_points=np.array(reference_points),
                            model_name=model_name,
                            data_source=config.data_source,
                            data_name=data_name,
                            num_preferences=config.num_preferences_test,
                            num_sampling_cycles=config.num_sampling_cycles,
                            strategy_tag=strategy_tag,
                        )
                    scenario_results.append(result)
                    for metric_name, metric_values in performance_metrics.items():
                        aggregated_metrics[metric_name].append(np.mean(metric_values))

                scenario_results = np.array(scenario_results)
                save_result = np.mean(scenario_results, axis=0)
                save_std = np.std(scenario_results, axis=0)

                np.save(save_path, save_result)
                np.savez(
                    save_path.replace(".npy", ".npz"),
                    seeds=np.array(test_seeds),
                    per_seed_results=scenario_results,
                    mean=save_result,
                    std=save_std,
                )

                hv_values = scenario_results[:, :, 0]
                pareto_sizes = scenario_results[:, :, 1]
                runtime_values = scenario_results[:, :, 2]
                mode_name = "Sampling" if flag_sample else "Greedy"
                print("Test summary:")
                print(
                    f"Normalized HV ({mode_name}): {hv_values.mean():.6f} +/- {hv_values.std():.6f}"
                )
                print(
                    "Pareto set size:"
                    f" {pareto_sizes.mean():.2f} +/- {pareto_sizes.std():.2f}"
                )
                print(
                    f"Runtime (s): {runtime_values.mean():.4f} +/- {runtime_values.std():.4f}"
                )
                if aggregated_metrics:
                    metric_summary = {
                        metric_name: (
                            float(np.mean(metric_values)),
                            float(np.std(metric_values)),
                        )
                        for metric_name, metric_values in aggregated_metrics.items()
                    }
                    print(
                        "Performance metrics (mean +/- std across scenarios):"
                        f" {metric_summary}"
                    )


if __name__ == "__main__":
    main(configs, configs.test_mode)
