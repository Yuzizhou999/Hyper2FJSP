import argparse
import json
import logging
import multiprocessing
import os
import time
from multiprocessing.pool import Pool

import numpy as np
from deap import base, creator, tools
from genetic_algorithm.operators import (
    ObjectiveFn,
    evaluate_individual,
    evaluate_population,
    init_individual,
    init_population,
    mutate_sequence_exchange,
    mutate_shortest_proc_time,
    pox_crossover,
    repair_precedence_constraints,
    variation,
)
from helper_functions import (
    load_job_shop_env,
    load_parameters,
    record_stats,
    save_results,
)
from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from utils import (
    REFERENCE_POINT_INDICES,
    compute_hypervolume,
    get_earliness_reference_index,
)

logging.basicConfig(level=logging.INFO)

from config import DEFAULT_RESULTS_ROOT, REFERENCE_POINTS_FILE  # noqa: E402

PARAM_FILE = "configs/genetic_algorithm.json"


def initialize_run(pool: Pool, **kwargs):
    """Initializes the run by setting up the environment, toolbox, statistics, hall of fame, and initial population.

    Args:
        pool: Multiprocessing pool.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the initial population, toolbox, statistics, hall of fame, and environment.
    """
    try:
        jobShopEnv = load_job_shop_env(kwargs["problem_instance"])
    except FileNotFoundError:
        logging.error(f"Problem instance {kwargs['problem_instance']} not found.")
        return

    toolbox = base.Toolbox()
    # Allow DEAP to parallelize fitness evaluation when a pool is provided
    if pool is not None:
        toolbox.register("map", pool.map)
    # Register minimization fitness and individual; guards omitted because script runs standalone
    creator.create(
        "Fitness",
        base.Fitness,
        weights=tuple([-1.0 for _ in range(kwargs["nr_of_objectives"])]),
    )
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox.register(
        "init_individual",
        init_individual,
        creator.Individual,
        kwargs,
        jobShopEnv=jobShopEnv,
    )
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

    toolbox.register(
        "mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=jobShopEnv
    )
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selNSGA2)
    deadline_alpha = kwargs.get("deadline_alpha", 0.5)
    if "objective_names" in kwargs and kwargs["objective_names"] is not None:
        toolbox.register(
            "evaluate_individual",
            evaluate_individual,
            jobShopEnv=jobShopEnv,
            objectives=kwargs["nr_of_objectives"],
            objective_names=kwargs["objective_names"],
            deadline_alpha=deadline_alpha,
        )
    else:
        toolbox.register(
            "evaluate_individual",
            evaluate_individual,
            jobShopEnv=jobShopEnv,
            objectives=kwargs["nr_of_objectives"],
            deadline_alpha=deadline_alpha,
        )

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    initial_population = init_population(
        toolbox,
        kwargs["population_size"],
    )
    try:
        fitnesses = evaluate_population(
            toolbox, initial_population, kwargs["nr_of_objectives"], logging
        )
    except Exception as e:
        logging.error(f"An error occurred during initial population evaluation: {e}")
        return

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof, jobShopEnv


def run_algo(
    jobShopEnv,
    population,
    toolbox,
    folder,
    exp_name,
    stats=None,
    hof=None,
    objective_lbs=None,
    **kwargs,
):
    """Executes the genetic algorithm and returns the best individual.

    Args:
        jobShopEnv: The problem environment.
        population: The initial population.
        toolbox: DEAP toolbox.
        folder: The folder to save results in.
        exp_name: The experiment name.
        stats: DEAP statistics (optional).
        hof: Hall of Fame (optional).
        kwargs: Additional keyword arguments.

    Returns:
        The best individual found by the genetic algorithm.
    """

    if kwargs["plotting"]:
        draw_precedence_relations(jobShopEnv)  # Visualize graph once at start

    hof.update(population)

    gen = 0
    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs["logbook"], df_list, logging)

    if kwargs["logbook"]:
        logging.info(logbook.stream)

    # Start timing from the first generation
    start_time = time.time()

    for gen in range(1, kwargs["ngen"] + 1):
        # Vary the population
        offspring = variation(
            population,
            toolbox,
            kwargs["population_size"],
            kwargs["cr"],
            kwargs["indpb"],
        )

        # Ensure precedence constraints between jobs are satisfied (assembly scheduling cases)
        if "/dafjs/" or "/yfjs/" in jobShopEnv.instance_name:
            offspring = repair_precedence_constraints(jobShopEnv, offspring)

        # Evaluate the offspring
        fitnesses = evaluate_population(
            toolbox, offspring, kwargs["nr_of_objectives"], logging
        )
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select next generation population
        population[:] = toolbox.select(
            population + offspring, kwargs["population_size"]
        )
        # Update the statistics with the new population
        record_stats(
            gen, population, logbook, stats, kwargs["logbook"], df_list, logging
        )

    # End timing after the last generation
    end_time = time.time()
    runtime_seconds = end_time - start_time
    kwargs["runtime_seconds"] = runtime_seconds
    logging.info(f"Total runtime: {runtime_seconds:.2f} seconds")

    # Load existing reference point and compute hypervolume
    if os.path.isfile(REFERENCE_POINTS_FILE):
        with open(REFERENCE_POINTS_FILE, "r") as file:
            reference_points = json.load(file)
            if kwargs["problem_instance"] in reference_points:
                if (
                    kwargs["objective_names"] is not None
                    and len(kwargs["objective_names"]) > 0
                ):
                    objectives_functions = [
                        ObjectiveFn(str(obj).lower())
                        for obj in kwargs["objective_names"]
                    ]
                    deadline_alpha = kwargs.get("deadline_alpha", 0.5)
                    indices = [
                        get_earliness_reference_index(deadline_alpha)
                        if obj == ObjectiveFn.TOTAL_EARLINESS
                        else REFERENCE_POINT_INDICES[obj]
                        for obj in objectives_functions
                    ]
                    reference_point = [
                        reference_points[kwargs["problem_instance"]][i] for i in indices
                    ]
                    lower_bounds = [objective_lbs[obj] for obj in objectives_functions]
                else:
                    reference_point = [
                        reference_points[kwargs["problem_instance"]][i] for i in [0, -2]
                    ]  # [0:kwargs['nr_of_objectives']]
                hypervolume = compute_hypervolume(
                    hof, kwargs["nr_of_objectives"], list(reference_point), lower_bounds
                )
                kwargs["hypervolume"] = hypervolume
            else:
                print("NO REFERENCE POINT KNOWN")

    if kwargs["plotting"]:
        deadline_alpha = kwargs.get("deadline_alpha", 0.5)
        objectives, jobShopEnv = evaluate_individual(
            hof[0],
            jobShopEnv,
            kwargs["nr_of_objectives"],
            objective_names=kwargs.get("objective_names"),
            reset=False,
            deadline_alpha=deadline_alpha,
        )
        draw_gantt_chart(jobShopEnv)

    if folder is not None:
        save_results(hof, logbook, folder, exp_name, kwargs)

    return hypervolume


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    pool = multiprocessing.Pool()

    # Create a specific folder for the objective function combination if specified
    objectives_subfolder = ""
    if "objective_names" in parameters and parameters["objective_names"]:
        objectives_subfolder = "/" + "_".join(parameters["objective_names"])

    folder = (
        DEFAULT_RESULTS_ROOT
        + "/GA"  # Add algorithm-specific folder
        + "/"
        + str(parameters["problem_instance"])
        + objectives_subfolder  # Add the objectives subfolder
        + "/ngen"
        + str(parameters["ngen"])
        + "_pop"
        + str(parameters["population_size"])
        + "_cr"
        + str(parameters["cr"])
        + "_indpb"
        + str(parameters["indpb"])
    )

    exp_name = "rseed" + str(parameters["rseed"])
    population, toolbox, stats, hof, jobShopEnv = initialize_run(pool, **parameters)

    makespan_lb = max(
        [
            sum(
                min(operation.processing_times.values()) for operation in job.operations
            )
            for job in jobShopEnv.jobs
        ]
    )
    flowtime_lb = np.mean(
        [
            sum(
                min(operation.processing_times.values()) for operation in job.operations
            )
            for job in jobShopEnv.jobs
        ]
    )
    tardiness_lb = 0
    earliness_lb = 0
    costs_lb = sum(
        sum(
            jobShopEnv.max_duration - max(operation.processing_times.values())
            for operation in job.operations
        )
        for job in jobShopEnv.jobs
    )
    objective_lbs = {
        ObjectiveFn.MAKESPAN: makespan_lb,
        ObjectiveFn.AVERAGE_FLOWTIME: flowtime_lb,
        ObjectiveFn.TOTAL_TARDINESS: tardiness_lb,
        ObjectiveFn.TOTAL_EARLINESS: earliness_lb,
        ObjectiveFn.COSTS: costs_lb,
    }

    run_algo(
        jobShopEnv,
        population,
        toolbox,
        folder,
        exp_name,
        stats,
        hof,
        objective_lbs,
        **parameters,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
