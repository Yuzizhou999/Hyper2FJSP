import copy
import os
from glob import glob
from typing import Dict, Optional

import numpy as np

from data_utils import pack_data_from_config
from enums import ObjectiveFn
from hypervolume_utils import compute_hypervolume
from mo_fjsp_env_various_op_nums import MOFJSPEnvForVariousOpNums
from params import configs


def find_pareto_efficient_solutions(points: np.ndarray) -> np.ndarray:
    """
    Find Pareto-efficient solutions from a set of points.

    Args:
        points: A 2D numpy array where each row is a point in the objective space.

    Returns:
        A 2D numpy array containing the Pareto-efficient points.
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)  # Keep any point with a lower value
            is_efficient[i] = True  # And keep self
    return points[is_efficient]

def read_or_tools_solutions(
    data_source: Optional[str] = None,
    test_data: Optional[list] = None,
    objective_fn: Optional[list[str]] = None,
    base_dir: str = "or_solution",
    ) -> Dict[str, np.ndarray]:
    """
    Read all solutions for all weight combinations for all instances and compute normalized hypervolumes.

    Args:
        data_source: The data source directory name (corresponds to configs.data_source).
                     If None, uses configs.data_source.
        test_data: The test data directory name as a list with one element (corresponds to configs.test_data).
                   If None, uses configs.test_data.
        objective_fn: The objective function directory name as a list (e.g., ["makespan", "flowtime"]).
                      If None, uses configs.objective_fn.
        base_dir: The base directory containing the OR-Tools solutions.
        reference_points: Reference points for hypervolume calculation.
        objective_lower_bounds: Lower bounds for objectives for normalization.
        sensible_reference_points: Sensible reference points for hypervolume calculation.

    Returns:
        Dictionary mapping problem instances to arrays of points (coordinates) over all weight combinations,
        their corresponding normalized hypervolumes, and the number of unique coordinates in the Pareto sets.
    """
    # Use values from configs if not provided
    data_source = data_source if data_source is not None else configs.data_source
    test_data = test_data if test_data is not None else configs.test_data
    objective_fn = objective_fn if objective_fn is not None else configs.objective_fn

    # Join the objective function list into a single string
    objective_fn_str = "_".join(objective_fn)

    # Construct the path to the relevant solution subdirectory
    solution_dir = os.path.join(base_dir, data_source, objective_fn_str)

    # Check if the directory exists
    if not os.path.exists(solution_dir):
        raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

    # Find all weight subdirectories in the objective function directory
    weight_dirs = [d for d in glob(os.path.join(solution_dir, "*")) if os.path.isdir(d)]

    if not weight_dirs:
        print(f"Warning: No weight directories found in {solution_dir}")

    # Initialize a dictionary to store points, hypervolumes, and unique Pareto set sizes for each problem instance
    instance_points = {}
    instance_hypervolumes = {}
    instance_unique_pareto_sizes = {}
    second_axis_shape = None

    for weight_dir in weight_dirs:
        weight_name = os.path.basename(weight_dir)
        solution_file = os.path.join(weight_dir, f"solution_{test_data[0]}.npy")

        if not os.path.exists(solution_file):
            print(f"Warning: Solution file not found for weight {weight_name}: {solution_file}")
            continue

        try:
            solution_data = np.load(solution_file, allow_pickle=True)
            # Assert that the first dimension of the solution array is 100
            # assert solution_data.shape[0] == 100, (
            #     f"Solution array for weight {weight_name} does not have 100 instances: "
            #     f"found {solution_data.shape[0]}"
            # )
            # Assert that the second dimension is consistent across all solutions
            if second_axis_shape is None:
                second_axis_shape = solution_data.shape[1]
            else:
                assert solution_data.shape[1] == second_axis_shape, (
                    f"Solution array for weight {weight_name} has a different second axis shape: "
                    f"expected {second_axis_shape}, found {solution_data.shape[1]}"
                )

            # Extract coordinates (values after the first 2 values)
            coordinates = solution_data[:, 2:]

            # Add coordinates to the corresponding problem instance
            for i, point in enumerate(coordinates):
                if i not in instance_points:
                    instance_points[i] = []
                instance_points[i].append(point)

        except Exception as e:
            print(f"Error loading {solution_file}: {e}")
    
    # collect the test data
    data, reference_points = pack_data_from_config(
        data_source, test_data, load_reference_points=True
    )
    reference_points = np.array(reference_points[0][0])
    
    # Reference points are loaded in the order: [makespan, average_flowtime, total_tardiness, 
    # total_earliness(0.5), total_earliness(0.9), total_earliness(1.0), costs]
    # (see data_utils.py line 166)
    # We need to map each objective to its position in the loaded reference_points array
    loaded_ref_order = {
        ObjectiveFn.MAKESPAN: 0,
        ObjectiveFn.AVERAGE_FLOWTIME: 1,
        ObjectiveFn.TOTAL_TARDINESS: 2,
        ObjectiveFn.COSTS: 6,
    }
    
    # Build indices list in the order of objective_fn
    reference_point_indices = []
    objective_fn_enums = list(map(lambda x: ObjectiveFn(x.lower()), objective_fn))
    
    for obj_fn in objective_fn_enums:
        if obj_fn in loaded_ref_order:
            reference_point_indices.append(loaded_ref_order[obj_fn])
        elif obj_fn == ObjectiveFn.TOTAL_EARLINESS:
            # Map deadline_alpha to appropriate earliness reference point index
            # Index 3: alpha=0.5, Index 4: alpha=0.9, Index 5: alpha=1.0
            if configs.deadline_alpha <= 0.51:
                reference_point_indices.append(3)  # alpha 0.5
            elif configs.deadline_alpha <= 0.91:
                reference_point_indices.append(4)  # alpha 0.9
            else:
                reference_point_indices.append(5)  # alpha 1.0
        elif obj_fn == ObjectiveFn.NEGATIVE_MAKESPAN:
            reference_point_indices.append(-1)
    
    reference_points = reference_points[:, reference_point_indices]
    
    
    
    ####### COMPUTE LOWER BOUNDS #######
    data = data[0][0]
    n_j = data[0][0].shape[0]
    n_m = data[1][0].shape[1]
    env = MOFJSPEnvForVariousOpNums(
        n_j=n_j, n_m=n_m, objective_fns=list(map(lambda x: ObjectiveFn(x.lower()), objective_fn)), data_source=data_source
    )
    if data_source != "BenchData":
        env.set_initial_data(
                data[0],
                data[1],
                deadline_alpha=configs.deadline_alpha,
            )
    
        # Convert lists of points to numpy arrays
        for instance in instance_points:
            instance_points[instance] = np.array(instance_points[instance])

            # Compute normalized hypervolume for each instance using sensible reference points
            pareto_set = find_pareto_efficient_solutions(instance_points[instance])
            normalized_hypervolume = compute_hypervolume(
                pareto_set - env.objective_lower_bounds[instance],
                reference_points[instance] - env.objective_lower_bounds[instance] + 1e-8,
            ) / np.prod(reference_points[instance] - env.objective_lower_bounds[instance] + 1e-8)
            instance_hypervolumes[instance] = normalized_hypervolume

            # Store the number of unique coordinates in the Pareto set
            unique_pareto_points = np.unique(pareto_set, axis=0)
            instance_unique_pareto_sizes[instance] = unique_pareto_points.shape[0]

    else:
        for instance in instance_points:
            instance_points[instance] = np.array(instance_points[instance])

            # Compute normalized hypervolume for each instance using sensible reference points
            pareto_set = find_pareto_efficient_solutions(instance_points[instance])
            env.set_initial_data(
                data[0][instance: instance + 1],
                data[1][instance: instance + 1],
                deadline_alpha=configs.deadline_alpha,
            )
            normalized_hypervolume = compute_hypervolume(
                pareto_set - env.objective_lower_bounds[0],
                reference_points[instance] - env.objective_lower_bounds[0] + 1e-8,
            ) / np.prod(reference_points[instance] - env.objective_lower_bounds[0] + 1e-8)
            instance_hypervolumes[instance] = normalized_hypervolume

            # Store the number of unique coordinates in the Pareto set
            unique_pareto_points = np.unique(pareto_set, axis=0)
            instance_unique_pareto_sizes[instance] = unique_pareto_points.shape[0]

    # Assert the number of weight combinations based on the second axis shape
    if second_axis_shape == 4:
        assert len(weight_dirs) == 101, (
            f"Expected 101 weight combinations for second axis shape 4, but found {len(weight_dirs)}"
        )
    elif second_axis_shape == 5:
        assert len(weight_dirs) == 105, (
            f"Expected 105 weight combinations for second axis shape 5, but found {len(weight_dirs)}"
        )
    elif second_axis_shape == 6:
        assert len(weight_dirs) == 120, (
            f"Expected 120 weight combinations for second axis shape 6 (4 objectives), but found {len(weight_dirs)}"
        )

    return instance_points, instance_hypervolumes, instance_unique_pareto_sizes

if __name__ == "__main__":
    # Example usage
    try:
        # Replace with actual data source, test data, and objective function names
        solutions, hypervolumes, unique_pareto_sizes = read_or_tools_solutions(
            data_source=configs.data_source,
            test_data=configs.test_data,
            objective_fn=configs.objective_fn, # Replace with actual reference points  # Replace with actual objective lower bounds
                    )
        print(f"Successfully loaded solutions for {solutions[0].shape[0]} weight combinations.")
        print(f"Number of objectives: {solutions[0].shape[1]}")
        print(f"Computed hypervolumes for {len(hypervolumes)} instances.")
        print(f"Computed unique Pareto sizes for {len(unique_pareto_sizes)} instances.")

        # Calculate and print average hypervolume and average Pareto sizes
        avg_hypervolume = np.mean(list(hypervolumes.values()))
        avg_pareto_size = np.mean(list(unique_pareto_sizes.values()))
        print(f"Average Hypervolume: {avg_hypervolume}")
        print(f"Average Pareto Size: {avg_pareto_size}")

    except Exception as e:
        raise e
