import collections
import itertools
import os
import sys
import time

import numpy as np
from ortools.sat.python import cp_model
from tqdm import tqdm

from data_utils import pack_data_from_config
from enums import ObjectiveFn
from multipliers import MULTIPLIERS, MULTIPLIERS_BenchData
from params import configs
from common_utils import das_dennis

print(f"TASK ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
    # SLURM array index selects the weight vector so each task solves a distinct preference point
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    assert isinstance(configs.objective_fn, (list, tuple))
    if len(configs.objective_fn) == 2:
        potential_weights = np.linspace(0, 1, 101)
        potential_weights = np.array([potential_weights, 1 - potential_weights]).T
    elif len(configs.objective_fn) == 3:
        potential_weights = das_dennis(13, 3)
    elif len(configs.objective_fn) == 4:
        potential_weights = das_dennis(7, 4)  # 120 preference vectors
    weights = list(potential_weights[task_id])
    print(f"Using weights: {weights}")
    configs.weights = weights
else:
    task_id = configs.task_id
    assert isinstance(configs.objective_fn, (list, tuple))
    if len(configs.objective_fn) == 2:
        potential_weights = np.linspace(0, 1, 101)
        potential_weights = np.array([potential_weights, 1 - potential_weights]).T
    elif len(configs.objective_fn) == 3:
        potential_weights = das_dennis(13, 3)
    elif len(configs.objective_fn) == 4:
        potential_weights = das_dennis(7, 4)  # 120 preference vectors
    weights = list(potential_weights[task_id])
    print(f"Using weights: {weights}")
    configs.weights = weights


os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def solve_instances(config):
    """
        Solve 'test_data' from 'data_source' using OR-Tools
        with time limits 'max_solve_time' for each instance,
        and save the result to './or_solution/{data_source}'
    :param config: a package of parameters
    :return:
    """

    # Handle objective_fn as either a single objective or a list of objectives
    if isinstance(config.objective_fn, (list, tuple)):
        # Multiple objectives
        objective_fns = [ObjectiveFn(obj) for obj in config.objective_fn]

        # Check for weights, if not provided use equal weights
        if (
            hasattr(config, "weights")
            and isinstance(config.weights, (list, tuple))
            and len(config.weights) == len(objective_fns)
        ):
            weights = config.weights
        else:
            weights = [1.0] * len(objective_fns)
            print(f"Using equal weights for {len(objective_fns)} objectives")
    else:
        # Single objective (backward compatibility)
        objective_fns = [ObjectiveFn(config.objective_fn)]
        weights = [1.0]

    if not os.path.exists(f"./or_solution/{config.data_source}"):
        os.makedirs(f"./or_solution/{config.data_source}")

    data_list = pack_data_from_config(config.data_source, config.test_data)

    # Create folder structure:
    # ./or_solution/{data_source}/{objectives}/{weights}/{data_name}/solution_{data_name}_{instance_id}.npy

    # 1. First create objective names subfolder
    if isinstance(config.objective_fn, (list, tuple)):
        objective_names = [
            str(ObjectiveFn(obj)).split(".")[-1].lower() for obj in config.objective_fn
        ]
        objectives_str = "_".join(objective_names)
    else:
        objective_name = str(ObjectiveFn(config.objective_fn)).split(".")[-1].lower()
        objectives_str = objective_name

    # 2. Then create weights subfolder
    if isinstance(config.weights, (list, tuple)):
        weights_str = "_".join([f"{w}" for w in weights])
    else:
        weights_str = "1.0"

    # Construct path with appropriate nesting
    save_dir_objectives = f"./or_solution/{config.data_source}/{objectives_str}"
    save_dir_weights = f"{save_dir_objectives}/{weights_str}"

    # Create objective and weights directories
    if not os.path.exists(save_dir_objectives):
        os.makedirs(save_dir_objectives, exist_ok=True)
    if not os.path.exists(save_dir_weights):
        os.makedirs(save_dir_weights, exist_ok=True)

    # Print information about objectives being optimized
    if isinstance(config.objective_fn, (list, tuple)):
        objective_names = [
            str(ObjectiveFn(obj)).split(".")[-1] for obj in config.objective_fn
        ]
        print(
            f"Optimizing multiple objectives: {objective_names} with weights: {weights}"
        )
    else:
        print(
            f"Optimizing single objective: {str(ObjectiveFn(config.objective_fn)).split('.')[-1]}"
        )

    for data in data_list:
        dataset = data[0]
        data_name = data[1]

        # Set up paths for the problem instance subfolder
        save_path = f"{save_dir_weights}/solution_{data_name}.npy"
        save_subpath = f"{save_dir_weights}/{data_name}"

        if not os.path.exists(save_subpath):
            os.makedirs(save_subpath, exist_ok=True)

        if True:  # (not os.path.exists(save_path)) or config.cover_flag:
            print("-" * 25 + "Solve Setting" + "-" * 25)
            print(f"solve data name : {data_name}")
            print(f"path : ./data/{config.data_source}/{data_name}")
            print(f"objectives: {objectives_str}")
            print(f"weights: {weights_str}")
            print(f"saving to: {save_subpath}")

            # search for the start index
            # for root, dirs, files in os.walk(save_subpath):
            #     index = len([int(f.split("_")[-1][:-4]) for f in files])

            print(f"left instances: dataset[, {len(dataset[0])})")
            for k in tqdm(
                range(len(dataset[0])), file=sys.stdout, desc="progress", colour="blue"
            ):
                jobs, num_machines = matrix_to_the_format_for_solving(
                    dataset[0][k], dataset[1][k]
                )
                if (
                    ObjectiveFn.TOTAL_TARDINESS in objective_fns
                    or ObjectiveFn.TOTAL_EARLINESS in objective_fns
                ):
                    deadlines = generate_deadlines(
                        dataset[0][k], dataset[1][k], slack_factor=config.deadline_alpha
                    )
                else:
                    deadlines = None
                solution, solveTime, individual_values = fjsp_solver(
                    jobs=jobs,
                    num_machines=num_machines,
                    time_limits=config.max_solve_time,
                    objective_fn=objective_fns,
                    weights=weights,
                    deadlines=deadlines,
                    data_source=config.data_source,
                )

                # Save both aggregated and individual objective values
                result_to_save = np.array([solution, solveTime, *individual_values])

                # Format each objective value with its name for clearer output
                if isinstance(objective_fns, list) and len(objective_fns) > 1:
                    obj_names = [str(obj).split(".")[-1] for obj in objective_fns]
                    obj_values_str = ", ".join(
                        [
                            f"{name}: {value}"
                            for name, value in zip(obj_names, individual_values)
                        ]
                    )
                    tqdm.write(
                        f"Instance {k + 1}, weighted sum: {solution}, individual objectives: [{obj_values_str}], solveTime: {solveTime:.2f}s, systemtime: {time.strftime('%m-%d %H:%M:%S')}"
                    )
                else:
                    tqdm.write(
                        f"Instance {k + 1}, solution: {solution}, solveTime: {solveTime:.2f}s, systemtime: {time.strftime('%m-%d %H:%M:%S')}"
                    )

                np.save(
                    save_subpath
                    + f"/solution_{data_name}_{str.zfill(str(k + 1), 3)}.npy",
                    result_to_save,
                )

            print("load results...")
            results = []
            for i in range(len(dataset[0])):
                solve_msg = np.load(
                    save_subpath
                    + f"/solution_{data_name}_{str.zfill(str(i + 1), 3)}.npy"
                )
                results.append(solve_msg)

            np.save(save_path, np.array(results))
            print("successfully save results...")


def matrix_to_the_format_for_solving(job_length, op_pt):
    """
        Convert matrix form of the data into the format needed by OR-Tools
    :param job_length: the number of operations in each job (shape [J])
    :param op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    :return:
    """
    num_ops, num_machines = op_pt.shape
    num_jobs = job_length.shape[0]
    jobs = []
    op_idx = 0
    for j in range(num_jobs):
        job_msg = []
        for k in range(job_length[j]):
            able_mchs = np.where(op_pt[op_idx] != 0)[0]
            op_msg = [(op_pt[op_idx, k], k) for k in able_mchs]
            job_msg.append(op_msg)
            op_idx += 1
        jobs.append(job_msg)
    return jobs, num_machines


def generate_deadlines(job_length, op_pt, slack_factor=1.0):
    """
    Generate deadlines for a set of jobs based on their operation processing times and a slack factor.
    This function calculates a deadline for each job by summing the minimum processing time
    of each operation (ignoring non-positive time values) for that job and then adding a slack
    margin proportional to the total computed minimum processing time.
    Parameters:
        job_length (List[int]): A list where each element represents the number of operations in a job.
        op_pt (Iterable): An iterable (e.g., list or array) of processing times for the operations.
                          Each element is expected to be indexable (e.g., a NumPy array) so that the
                          condition op_pt[i] > 0 can be applied to filter valid processing times.
        slack_factor (float, optional): A multiplier used to compute additional slack time for each job.
                                        The slack is calculated as slack_factor multiplied by the sum
                                        of minimum processing times for all operations in the job.
                                        Defaults to 0.5.
    Returns:
        List[float]: A list of deadlines where each deadline is the sum of the minimum operation
                     processing times for a job plus the corresponding slack.
    """

    num_jobs = len(job_length)
    deadlines = []
    op_idx = 0
    for j in range(num_jobs):
        job_deadline = 0
        for _ in range(job_length[j]):
            valid_times = op_pt[op_idx][op_pt[op_idx] > 0]
            if valid_times.size > 0:
                min_time = int(valid_times.min())
            else:
                min_time = 0
            job_deadline += min_time
            op_idx += 1
        slack = slack_factor * job_deadline
        deadlines.append(job_deadline + slack)
    return deadlines


def fjsp_solver(
    jobs,
    num_machines,
    time_limits,
    objective_fn,
    deadlines=None,
    weights=None,
    data_source=None,
):
    """
        solve a fjsp instance by OR-Tools
        (imported from https://github.com/google/or-tools/blob/master/examples/python/flexible_job_shop_sat.py)
    :param jobs: a list of processing information
    :param num_machines: the number of machines
    :param time_limits: the time limits for solving the instance
    :param objective_fn: single objective or list of objectives to optimize
    :param deadlines: deadlines for jobs when using tardiness objective
    :param weights: weights for each objective in weighted sum (used only when objective_fn is a list)
    :return: tuple of (objective value, solve time, individual objective values)
    """

    num_jobs = len(jobs)
    if data_source == "BenchData":
        multipliers = MULTIPLIERS_BenchData
    else:
        multipliers = MULTIPLIERS[(data_source, num_jobs, num_machines)]
    all_jobs = range(num_jobs)

    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = "_j%i_t%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(
                min_duration, max_duration, "duration" + suffix_name
            )
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(
                start, duration, end, "interval" + suffix_name
            )

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar("presence" + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                    )
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Handle both single objective and multiple objectives for weighted sum
    if isinstance(objective_fn, list):
        objectives = objective_fn
        # If weights not provided, use equal weights
        if weights is None:
            weights = [1.0] * len(objectives)
        elif len(weights) != len(objectives):
            print(
                f"Warning: Number of weights ({len(weights)}) doesn't match objectives ({len(objectives)}). Using equal weights."
            )
            weights = [1.0] * len(objectives)
    else:
        # Single objective (backward compatibility)
        objectives = [objective_fn]
        weights = [1.0]

    # Create variables for each objective
    objective_vars = []

    # Define each objective
    for i, obj in enumerate(objectives):
        # Makespan objective
        if obj == ObjectiveFn.MAKESPAN:
            makespan = model.NewIntVar(0, horizon, f"makespan_{i}")
            model.AddMaxEquality(makespan, job_ends)
            objective_vars.append((makespan, weights[i] * multipliers[obj]))

        # Negative makespan objective
        elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
            makespan = model.NewIntVar(0, horizon, f"neg_makespan_{i}")
            model.AddMaxEquality(makespan, job_ends)
            # Negative weight to maximize instead of minimize
            objective_vars.append((makespan, -weights[i] * multipliers[obj]))

        # Average flowtime objective
        elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
            flowtime = model.NewIntVar(0, horizon * num_jobs, f"flowtime_{i}")
            model.Add(sum(job_ends[j] - starts[(j, 0)] for j in all_jobs) == flowtime)
            average_flowtime = model.NewIntVar(
                0, horizon * num_jobs, f"average_flowtime_{i}"
            )
            model.AddDivisionEquality(average_flowtime, flowtime, num_jobs)
            objective_vars.append((average_flowtime, weights[i] * multipliers[obj]))

        # Total tardiness objective
        elif obj == ObjectiveFn.TOTAL_TARDINESS:
            if deadlines is None:
                raise ValueError("Deadlines required for tardiness objective")

            tardiness_vars = []
            for j in all_jobs:
                deadline_val = int(round(deadlines[j]))
                tardiness = model.NewIntVar(0, horizon, f"tardiness_{i}_{j}")
                model.Add(tardiness >= job_ends[j] - deadline_val)
                model.Add(tardiness >= 0)
                tardiness_vars.append(tardiness)

            total_tardiness = model.NewIntVar(
                0, horizon * num_jobs, f"total_tardiness_{i}"
            )
            model.Add(total_tardiness == sum(tardiness_vars))
            objective_vars.append((total_tardiness, weights[i] * multipliers[obj]))

        # Total earliness objective
        elif obj == ObjectiveFn.TOTAL_EARLINESS:
            if deadlines is None:
                raise ValueError("Deadlines required for earliness objective")

            earliness_vars = []
            for j in all_jobs:
                deadline_val = int(round(deadlines[j]))
                earliness = model.NewIntVar(0, horizon, f"earliness_{i}_{j}")
                # Earliness = max(0, deadline - completion_time)
                model.Add(earliness >= deadline_val - job_ends[j])
                model.Add(earliness >= 0)
                earliness_vars.append(earliness)

            total_earliness = model.NewIntVar(
                0, horizon * num_jobs, f"total_earliness_{i}"
            )
            model.Add(total_earliness == sum(earliness_vars))
            objective_vars.append((total_earliness, weights[i] * multipliers[obj]))

        # Costs objective
        elif obj == ObjectiveFn.COSTS:
            max_processing_time = max(
                max(max(job[op_id], key=lambda x: x[0])[0] for op_id in range(len(job)))
                for job in jobs
            )
            costs_upper_bound = sum(
                sum(
                    max_processing_time - min(job[op_id], key=lambda x: x[0])[0]
                    for op_id in range(len(job))
                )
                for job in jobs
            )
            costs = model.NewIntVar(0, costs_upper_bound, f"costs_{i}")
            model.Add(
                costs
                == sum(
                    itertools.chain.from_iterable(
                        itertools.chain.from_iterable(
                            list(
                                list(
                                    list(
                                        (max_processing_time - x[0])
                                        * presences[(job_id, op_id, alt_id)]
                                        for alt_id, x in enumerate(job[op_id])
                                    )
                                    for op_id in range(len(job))
                                )
                                for job_id, job in enumerate(jobs)
                            )
                        )
                    )
                )
            )
            objective_vars.append((costs, weights[i] * multipliers[obj]))

    # Create weighted sum objective if there are multiple objectives or use single objective
    if len(objective_vars) > 1:
        weighted_sum_terms = []
        for var, weight in objective_vars:
            if weight != 0:  # Skip objectives with zero weight
                weighted_sum_terms.append(var * weight)

        if weighted_sum_terms:
            model.Minimize(sum(weighted_sum_terms))
    elif len(objective_vars) == 1:
        # Single objective
        obj_var, weight = objective_vars[0]
        if weight >= 0:
            model.Minimize(obj_var)
        else:
            model.Maximize(obj_var)
    else:
        raise ValueError("No valid objectives specified")

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limits
    solution_printer = SolutionPrinter()

    total1 = time.time()
    status = solver.Solve(model, solution_printer)
    total2 = time.time()

    # Extract individual objective values
    individual_values = [solver.Value(var) for var, _ in objective_vars]

    return solver.ObjectiveValue(), total2 - total1, individual_values


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """
    Print intermediate solutions.
    """

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """
        Called at each new solution.
        """
        print(
            "Solution %i, time = %f s, objective = %i"
            % (self.__solution_count, self.WallTime(), self.ObjectiveValue())
        )
        self.__solution_count += 1


if __name__ == "__main__":
    solve_instances(config=configs)
