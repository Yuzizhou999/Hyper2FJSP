import copy
import os
import sys
import time
from collections import defaultdict

import hvwfg
import numpy as np
import torch
from tqdm import tqdm

from common_utils import (
    das_dennis,
    find_pareto_efficient_solutions,
    greedy_select_action,
    sample_action,
    setup_seed,
)
from data_utils import pack_data_from_config
from enums import ObjectiveFn
from mo_fjsp_env_same_op_nums import MOFJSPEnvForSameOpNums
from model.PPO import PPO_initialize
from params import configs

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


device = torch.device(configs.device)

ppo = PPO_initialize(multi_objective=True)
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def save_pareto_sets(pareto_sets, data_source, data_name, model_name, strategy):
    """
    Save Pareto sets to disk

    Args:
        pareto_sets: List of Pareto sets for each instance
        data_source: Source of the data (e.g., 'SD1', 'SD2')
        data_name: Name of the dataset
        model_name: Name of the model
        strategy: Strategy used ('greedy' or 'sampling')
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
):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """

    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))
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
    )

    performance_metrics = defaultdict(list)

    if len(objective_fn) == 2:
        preferences = np.linspace(0, 1, num_preferences)
        preferences = np.array([preferences, 1 - preferences]).T
        preferences = torch.from_numpy(preferences).float().to(device)
    elif len(objective_fn) == 3:
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
        raise NotImplementedError("Unsupported number of objectives.")
    original_data_set_size = len(data_set[0])
    data_set = (
        # list(np.repeat(data_set[0], num_preferences, axis=0))
        [i for i in data_set[0] for _ in range(num_preferences)],
        [i for i in data_set[1] for _ in range(num_preferences)],
    )

    pareto_sets = []

    # For HYPER_DANIEL: generate parameters once for all unique preferences
    if hasattr(ppo.policy, "assign_preferences"):
        ppo.policy.assign_preferences(preferences)
        prefs_to_pass = None
    else:
        prefs_to_pass = preferences

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
            with torch.no_grad():
                pi, _ = ppo.policy(
                    fea_j=state.fea_j_tensor,  # [num_preferences, N, 8]
                    op_mask=state.op_mask_tensor,  # [num_preferences, N, N]
                    candidate=state.candidate_tensor,  # [num_preferences, J]
                    fea_m=state.fea_m_tensor,  # [num_preferences, M, 6]
                    mch_mask=state.mch_mask_tensor,  # [num_preferences, M, M]
                    comp_idx=state.comp_idx_tensor,  # [num_preferences, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor,
                    preferences=prefs_to_pass,  # None for HYPER_DANIEL, preferences for others
                )  # [num_preferences, J, M]

            action = greedy_select_action(pi)
            state, reward, done = env.step(actions=action.cpu().numpy())
            if done.all():
                break
        t2 = time.time()
        env.current_makespan[0]
        eval_objectives = []
        reference_point_indices = []
        lb_indices = []  # Track which index in objective_lower_bounds corresponds to each eval_objective
        if ObjectiveFn.MAKESPAN in objective_fn:
            eval_objectives.append(env.current_makespan)
            reference_point_indices.append(0)
            lb_indices.append(env.objectives.index(ObjectiveFn.MAKESPAN))
        if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn:
            eval_objectives.append(env.compute_avg_flowtime())
            reference_point_indices.append(1)
            lb_indices.append(env.objectives.index(ObjectiveFn.AVERAGE_FLOWTIME))
        if ObjectiveFn.TOTAL_TARDINESS in objective_fn:
            eval_objectives.append(env.compute_total_tardiness())
            reference_point_indices.append(2)
            lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_TARDINESS))
        if ObjectiveFn.TOTAL_EARLINESS in objective_fn:
            eval_objectives.append(env.compute_total_earliness())
            # Map deadline_alpha to appropriate earliness reference point index
            # Index 3: alpha=0.5, Index 4: alpha=0.9, Index 5: alpha=1.0
            if configs.deadline_alpha <= 0.51:
                reference_point_indices.append(3)  # alpha 0.5
            elif configs.deadline_alpha <= 0.91:
                reference_point_indices.append(4)  # alpha 0.9
            else:
                reference_point_indices.append(5)  # alpha 1.0
            lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_EARLINESS))
        if ObjectiveFn.COSTS in objective_fn:
            eval_objectives.append(env.compute_costs())
            reference_point_indices.append(len(reference_points[0]) - 1)
            lb_indices.append(env.objectives.index(ObjectiveFn.COSTS))
        normalization_values = copy.deepcopy(reference_point_indices)
        if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
            eval_objectives.append(-env.current_makespan)
            reference_point_indices.append(-1)
            normalization_values.append(0)
            lb_indices.append(env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN))

        eval_objectives = np.stack(eval_objectives, axis=1)
        eval_objectives = eval_objectives.reshape(1, num_preferences, -1)
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
            normalized_hypervolume = hvwfg.wfg(
                pareto_set - lower_bounds,
                reference_points[i][reference_point_indices] - lower_bounds + 1e-8,
            ) / np.prod(reference_points[i][normalization_values] - lower_bounds + 1e-8)
            normalized_hypervolumes.append(normalized_hypervolume)
            num_solutions_in_pareto_set = len(pareto_set)
        normalized_hypervolumes = np.array(normalized_hypervolumes)
        test_result_list.append(
            [normalized_hypervolume, num_solutions_in_pareto_set, t2 - t1]
        )

    # Save the Pareto sets if model_name is provided
    if model_name and data_source and data_name:
        save_pareto_sets(pareto_sets, data_source, data_name, model_name, "greedy")

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
):
    test_result_list = []
    all_cycles_pareto_sets = []  # Store pareto sets for each cycle
    all_cycles_performance_metrics = []  # Store performance metrics for each cycle

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))
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
    original_data_set_size = len(data_set[0])
    data_set = (
        [i for i in data_set[0] for _ in range(num_preferences * sample_times)],
        [i for i in data_set[1] for _ in range(num_preferences * sample_times)],
    )

    # For HYPER_DANIEL: generate parameters once for all unique preferences
    # Note: preferences tensor is repeated sample_times, but we only need unique ones
    # preferences.repeat(sample_times, axis=0) creates: [p0,p0,...,p0, p1,p1,...,p1, ...]
    # So we need to extract every sample_times-th element to get unique preferences
    if hasattr(ppo.policy, "assign_preferences"):
        # Extract unique preferences by taking every sample_times-th element
        unique_preferences = preferences[
            ::sample_times
        ]  # Takes indices 0, sample_times, 2*sample_times, ...
        ppo.policy.assign_preferences(unique_preferences)
        use_preference_indices = True
    else:
        use_preference_indices = False

    # Process each instance with all cycles before moving to next instance
    for i in tqdm(
        range(original_data_set_size),
        file=sys.stdout,
        desc="Instance progress",
        colour="blue",
    ):
        # Accumulate solutions across all cycles for this instance
        accumulated_solutions = []  # Store all solutions from all cycles

        # Run multiple sampling cycles for this instance
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

                        pi, _ = ppo.policy(
                            fea_j=state.fea_j_tensor,
                            op_mask=state.op_mask_tensor,
                            candidate=state.candidate_tensor,
                            fea_m=state.fea_m_tensor,
                            mch_mask=state.mch_mask_tensor,
                            comp_idx=state.comp_idx_tensor,
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                            fea_pairs=state.fea_pairs_tensor,
                            preferences=None,  # Already assigned
                            preference_indices=pref_indices,
                        )
                    else:
                        pi, _ = ppo.policy(
                            fea_j=state.fea_j_tensor,  # [num_preferences * sample_times, N, 8]
                            op_mask=state.op_mask_tensor,
                            candidate=state.candidate_tensor,
                            fea_m=state.fea_m_tensor,
                            mch_mask=state.mch_mask_tensor,
                            comp_idx=state.comp_idx_tensor,
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                            fea_pairs=state.fea_pairs_tensor,
                            preferences=preferences,
                        )

                action, _ = sample_action(pi)
                state, reward, done = env.step(actions=action.cpu().numpy())
                if done.all():
                    break
            t2 = time.time()
            env.current_makespan[0]
            eval_objectives = []
            reference_point_indices = []
            lb_indices = []  # Track which index in objective_lower_bounds corresponds to each eval_objective
            if ObjectiveFn.MAKESPAN in objective_fn:
                eval_objectives.append(env.current_makespan)
                reference_point_indices.append(0)
                lb_indices.append(env.objectives.index(ObjectiveFn.MAKESPAN))
            if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn:
                eval_objectives.append(env.compute_avg_flowtime())
                reference_point_indices.append(1)
                lb_indices.append(env.objectives.index(ObjectiveFn.AVERAGE_FLOWTIME))
            if ObjectiveFn.TOTAL_TARDINESS in objective_fn:
                eval_objectives.append(env.compute_total_tardiness())
                reference_point_indices.append(2)
                lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_TARDINESS))
            if ObjectiveFn.TOTAL_EARLINESS in objective_fn:
                eval_objectives.append(env.compute_total_earliness())
                # Map deadline_alpha to appropriate earliness reference point index
                # Index 3: alpha=0.5, Index 4: alpha=0.9, Index 5: alpha=1.0
                if configs.deadline_alpha <= 0.51:
                    reference_point_indices.append(3)  # alpha 0.5
                elif configs.deadline_alpha <= 0.91:
                    reference_point_indices.append(4)  # alpha 0.9
                else:
                    reference_point_indices.append(5)  # alpha 1.0
                lb_indices.append(env.objectives.index(ObjectiveFn.TOTAL_EARLINESS))
            if ObjectiveFn.COSTS in objective_fn:
                eval_objectives.append(env.compute_costs())
                reference_point_indices.append(len(reference_points[0]) - 1)
                lb_indices.append(env.objectives.index(ObjectiveFn.COSTS))
            normalization_values = copy.deepcopy(reference_point_indices)
            if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
                eval_objectives.append(-env.current_makespan)
                reference_point_indices.append(-1)
                normalization_values.append(0)
                lb_indices.append(env.objectives.index(ObjectiveFn.NEGATIVE_MAKESPAN))

            eval_objectives = np.stack(eval_objectives, axis=1)
            eval_objectives = eval_objectives.reshape(
                num_preferences * sample_times, -1
            )

            # Add solutions from this cycle to accumulated solutions
            accumulated_solutions.append(eval_objectives)

            # Combine all solutions from all cycles so far
            all_solutions_so_far = np.vstack(
                accumulated_solutions
            )  # Shape: [total_solutions_so_far, num_objectives]

            print(
                f"Cycle {cycle + 1}: Added {len(eval_objectives)} solutions. "
                f"Total accumulated: {len(all_solutions_so_far)} solutions. "
                f"Unique: {len(np.unique(all_solutions_so_far, axis=0))}"
            )

            # Compute Pareto set from all accumulated solutions
            pareto_set = find_pareto_efficient_solutions(all_solutions_so_far)
            print(f"Accumulated Pareto set size: {len(pareto_set)}")

            # Compute hypervolume of accumulated Pareto set
            # Reorder lower bounds to match the order of objectives using lb_indices
            all_lower_bounds = env.objective_lower_bounds[0]
            lower_bounds = all_lower_bounds[lb_indices]
            normalized_hypervolume = hvwfg.wfg(
                pareto_set - lower_bounds,
                reference_points[i][reference_point_indices] - lower_bounds + 1e-8,
            ) / np.prod(reference_points[i][normalization_values] - lower_bounds + 1e-8)
            num_solutions_in_pareto_set = len(pareto_set)

            print(f"Accumulated Normalized Hypervolume: {-normalized_hypervolume:.6f}")

            test_result_list.append(
                [normalized_hypervolume, num_solutions_in_pareto_set, t2 - t1]
            )

            # Store results for this instance and cycle
            all_cycles_pareto_sets.append(pareto_set)
            all_cycles_performance_metrics.append(performance_metrics)

            # Save the accumulated Pareto set for this instance and cycle
            if model_name and data_source and data_name:
                # Extract data from masked array if needed
                pareto_set_data = (
                    pareto_set.data if hasattr(pareto_set, "data") else pareto_set
                )

                if num_sampling_cycles > 1:
                    # Save with cycle number in the path
                    save_dir = f"./pareto_sets/{data_source}/{data_name}/{model_name}_sampling_cycle{cycle + 1}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/instance_{i}.npy"
                    np.save(save_path, pareto_set_data)
                else:
                    # Original behavior for single cycle
                    save_dir = (
                        f"./pareto_sets/{data_source}/{data_name}/{model_name}_sampling"
                    )
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = f"{save_dir}/instance_{i}.npy"
                    np.save(save_path, pareto_set_data)

    return np.array(test_result_list), performance_metrics


def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    objective_fn = [ObjectiveFn(str(obj_fn).lower()) for obj_fn in config.objective_fn]
    if not os.path.exists("./test_results"):
        os.makedirs("./test_results")

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append(
            (f"./trained_network/{config.model_source}/{model_name}.pth", model_name)
        )

    # collect the test data
    test_data, test_reference_points = pack_data_from_config(
        config.data_source, config.test_data, load_reference_points=True
    )

    if ObjectiveFn.NEGATIVE_MAKESPAN in objective_fn:
        test_reference_points[0] = (
            np.concatenate(
                (
                    np.array(test_reference_points[0][0]),
                    np.zeros((np.array(test_reference_points[0][0]).shape[0], 1)),
                ),
                axis=1,
            ),
            test_reference_points[0][1],
        )

    # Determine test mode
    if flag_sample:
        model_prefix = "DANIELS"
    else:
        model_prefix = "DANIELG"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f"./test_results/{config.data_source}/{data[1]}"
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f"/Result_{model_prefix}+{model[1]}_{data[1]}.npy"
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(1):
                        result, performance_metrics = test_greedy_strategy(
                            data[0],
                            model[0],
                            config.seed_test,
                            objective_fn=objective_fn,
                            reference_points=np.array(test_reference_points[0][0]),
                            model_name=model[1],
                            data_source=config.data_source,
                            data_name=data[1],
                            num_preferences=config.num_preferences_test,
                        )
                        # result, performance_metrics = test_sampling_strategy(data[0], model[0], 10, config.seed_test, objective_fn=objective_fn, reference_points=np.array(test_reference_points[0][0]))
                        result_5_times.append(result)
                    result_5_times = np.array(result_5_times)

                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print("Objective value (greedy): ", save_result[:, 0].mean())
                    print("Num solutions in pareto set: ", save_result[:, 1].mean())
                    print(
                        f"Performance metrics: {dict(zip(performance_metrics.keys(), map(np.mean, performance_metrics.values())))}"
                    )
                    print("time: ", save_result[:, 2].mean())

                else:
                    print("Test mode: Sampling")
                    result_5_times = []
                    # Sampling mode, test 5 times, record average time.
                    for j in range(1):
                        result, performance_metrics = test_sampling_strategy(
                            data[0],
                            model[0],
                            config.sample_times,
                            config.seed_test,
                            objective_fn=objective_fn,
                            reference_points=np.array(test_reference_points[0][0]),
                            model_name=model[1],
                            data_source=config.data_source,
                            data_name=data[1],
                            num_preferences=config.num_preferences_test,
                            num_sampling_cycles=config.num_sampling_cycles,
                        )
                        result_5_times.append(result)
                    result_5_times = np.array(result_5_times)

                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print("Objective value (sampling): ", save_result[:, 0].mean())
                    print("Num solutions in pareto set: ", save_result[:, 1].mean())
                    print(
                        f"Performance metrics: {dict(zip(performance_metrics.keys(), map(np.mean, performance_metrics.values())))}"
                    )
                    print("time: ", save_result[:, 2].mean())


if __name__ == "__main__":
    main(configs, configs.test_mode)
