import argparse
import json
import logging
import multiprocessing
import os
import random
import time
from multiprocessing.pool import Pool

import numpy as np
from config import DEFAULT_RESULTS_ROOT, REFERENCE_POINTS_FILE
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
)
from helper_functions import (
    load_job_shop_env,
    load_parameters,
    record_stats,
    save_results,
)
from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from pymoo.decomposition.tchebicheff import Tchebicheff
from utils import (
    REFERENCE_POINT_INDICES,
    compute_hypervolume,
    das_dennis,
    get_earliness_reference_index,
)

logging.basicConfig(level=logging.INFO)


# PARAM_FILE = "configs/moead.json"
PARAM_FILE = r"configs\moead\makespan_costs\BenchData\Brandimarte\1.json"


def initialize_run(pool: Pool, **kwargs):
    """Initializes the run by set up the environment, toolbox, statistics, hall of fame, and initial population.

    Args:
        pool: Multiprocessing pool.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the initial population, toolbox, statistics, hall of fame, and environment.
    """
    if kwargs["nr_of_objectives"] == 3:
        kwargs["population_size"] = 105
    elif kwargs["nr_of_objectives"] == 4:
        kwargs["population_size"] = 120  # das_dennis(7, 4) gives 120 preference vectors
    else:
        kwargs["population_size"] = 100
    try:
        jobShopEnv = load_job_shop_env(kwargs["problem_instance"])
    except FileNotFoundError:
        logging.error(f"Problem instance {kwargs['problem_instance']} not found.")
        return

    toolbox = base.Toolbox()
    if pool is not None:
        toolbox.register("map", pool.map)
    creator.create(
        "Fitness",
        base.Fitness,
        weights=tuple([-1.0 for i in range(kwargs["nr_of_objectives"])]),
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
    toolbox.register(
        "evaluate_individual",
        evaluate_individual,
        jobShopEnv=jobShopEnv,
        objectives=kwargs["nr_of_objectives"],
        objective_names=kwargs["objective_names"],
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


def assign_initial_population(population, weights, decomposition_method):
    """Assigns initial population to the individuals based on the weights and decomposition method.

    Args:
        population: The initial population.
        weights: The weights for the individuals.
        decomposition_method: The decomposition method to be used.

    Returns:
        The updated population with assigned weights.
    """
    # Compute minimum and maximum values for each objective
    min_vals = []
    max_vals = []
    for obj_nr in range(len(weights[0])):
        max_pop = max(
            population, key=lambda x: x.fitness.values[obj_nr]
        ).fitness.values[obj_nr]
        min_pop = min(
            population, key=lambda x: x.fitness.values[obj_nr]
        ).fitness.values[obj_nr]
        min_vals.append(min_pop)
        max_vals.append(max_pop)
    new_population = list(range(len(weights)))
    for i in np.random.permutation(range(len(weights))):
        decomposed_values = []
        for j in range(len(population)):
            normalized_vals = []
            for obj_nr in range(len(weights[i])):
                normalized_val = (
                    population[j].fitness.values[obj_nr] - min_vals[obj_nr]
                ) / (max_vals[obj_nr] - min_vals[obj_nr] + 1e-6)
                normalized_vals.append(normalized_val)
            decomposed_value = decomposition_method(normalized_vals, weights[i])
            decomposed_values.append(decomposed_value)
        # Find the index of the individual with the minimum decomposed value
        min_index = np.array(decomposed_values).flatten().argmin()
        # Append the individual with the minimum decomposed value to the new population
        new_population[i] = population.pop(min_index)

    return new_population


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
        draw_precedence_relations(jobShopEnv)  # Visualize precedence graph once
    decomposition_method = Tchebicheff()
    hof.update(population)

    gen = 0
    max_nr_replace = 2
    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs["logbook"], df_list, logging)

    if kwargs["logbook"]:
        logging.info(logbook.stream)

    # Preference/weight vectors drive decomposition
    weights = init_uniform_weights(
        kwargs["population_size"], kwargs["nr_of_objectives"]
    )

    population = assign_initial_population(population, weights, decomposition_method)

    NEIGHBORHOOD_SIZE = 20
    neighborhood = init_neighborhood(
        kwargs["population_size"], weights, NEIGHBORHOOD_SIZE
    )
    individuals, ideal_point, evaluations = init_ideal_point(
        kwargs["nr_of_objectives"], toolbox, kwargs["population_size"], population
    )
    PROB_NEIGHBORHOOD_MATING = 0.9  # 90% mating inside neighborhood; else global
    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs["logbook"], df_list, logging)
    if kwargs["logbook"]:
        logging.info(logbook.stream)

    evaluations = 0

    # Start timing from the first generation
    start_time = time.time()

    # for gen in range(1, kwargs['ngen'] + 1):
    while evaluations < kwargs["ngen"]:
        gen = evaluations
        for n in np.random.permutation(kwargs["population_size"]):
            rnd = np.random.random()
            if rnd < PROB_NEIGHBORHOOD_MATING:
                parents = np.random.choice(neighborhood[n], size=2, replace=False)
            else:
                parents = np.random.choice(
                    kwargs["population_size"], size=2, replace=False
                )

            # Apply crossover:

            ind1, ind2 = (
                toolbox.clone(population[parents[0]]),
                toolbox.clone(population[parents[1]]),
            )
            if random.random() < 0.5:
                ind1[0], ind2[0] = toolbox.mate_TwoPoint(ind1[0], ind2[0])
            else:
                ind1[0], ind2[0] = toolbox.mate_Uniform(ind1[0], ind2[0])
            ind1[1], ind2[1] = toolbox.mate_POX(ind1[1], ind2[1])
            del ind1.fitness.values, ind2.fitness.values

            # Apply mutation:
            # children = [ind1, ind2]
            children = [ind1, ind2]
            for child in children:
                child[0] = toolbox.mutate_machine_selection(child[0], 0.1)
                child[1] = toolbox.mutate_operation_sequence(child[1], 0.1)
                del child.fitness.values

            children = repair_precedence_constraints(jobShopEnv, children)

            offspring = []
            for child in children:
                fit = toolbox.evaluate_individual(child)[0]
                evaluations += 1
                child.fitness.values = fit
                offspring.append(child)

            pop_obj_vals = list(map(lambda x: x.fitness.values, population + offspring))
            min_vals = []
            max_vals = []
            for obj_nr in range(kwargs["nr_of_objectives"]):
                max_pop = max(pop_obj_vals, key=lambda x: x[obj_nr])[obj_nr]
                min_pop = min(pop_obj_vals, key=lambda x: x[obj_nr])[obj_nr]
                min_vals.append(min_pop)
                max_vals.append(max_pop)

            # Update the ideal point:
            for child in offspring:
                nr = 0
                # If the child has a better fitness value for any objective, update the ideal point:
                for j in range(kwargs["nr_of_objectives"]):
                    if child.fitness.values[j] < ideal_point[j]:
                        ideal_point[j] = child.fitness.values[j]
                        individuals[j] = child

                normalized_vals_child = []
                for obj_nr in range(kwargs["nr_of_objectives"]):
                    normalized_val_child = (
                        child.fitness.values[obj_nr] - min_vals[obj_nr]
                    ) / (max_vals[obj_nr] - min_vals[obj_nr] + 1e-6)
                    normalized_vals_child.append(normalized_val_child)
                # Update the solutions and problem:
                # Compute the Chebyshev decomposition of the objectives
                # decomposed_value = decomposition_method(normalized_vals, weights[n], utopian_point=ideal_point)
                # decomposed_value = decomposition_method(normalized_vals, weights[n])
                for old_individual_idx in np.random.permutation(neighborhood[n]):
                    normalized_vals = []
                    for obj_nr in range(kwargs["nr_of_objectives"]):
                        normalized_val = (
                            population[old_individual_idx].fitness.values[obj_nr]
                            - min_vals[obj_nr]
                        ) / (max_vals[obj_nr] - min_vals[obj_nr] + 1e-6)
                        normalized_vals.append(normalized_val)
                    # old_decomposed_value = decomposition_method(normalized_vals, weights[n], utopian_point=ideal_point)
                    decomposed_value = decomposition_method(
                        normalized_vals_child, weights[old_individual_idx]
                    )
                    old_decomposed_value = decomposition_method(
                        normalized_vals, weights[old_individual_idx]
                    )
                    if decomposed_value < old_decomposed_value:
                        population[old_individual_idx] = child
                        population[
                            old_individual_idx
                        ].fitness.values = child.fitness.values
                        nr += 1
                    if nr >= max_nr_replace:
                        break

        hof.update(population)

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


def init_ideal_point(n_objectives, toolbox, population_size, population):
    individuals = []
    ideal_point = np.zeros(n_objectives, dtype=float)
    evaluations = 0
    for i in range(n_objectives):
        individuals.append(toolbox.init_individual())
        individuals[i].fitness.values = toolbox.evaluate_individual(individuals[i])[0]
        ideal_point[i] = 1e30
        evaluations += 1  # TODO: Check if this is correct

    for i in range(population_size):
        for n in range(n_objectives):
            if population[i].fitness.values[n] < ideal_point[n]:
                ideal_point[n] = population[i].fitness.values[n]
                individuals[n] = population[i]
    return individuals, ideal_point, evaluations


def init_neighborhood(population_size, weights, neighborhood_size):
    x = np.empty(population_size)  # Of type float
    idx = [None] * population_size  # Of type int
    neighborhood = np.zeros((population_size, neighborhood_size), dtype=int)
    for i in range(population_size):
        for j in range(population_size):
            x[j] = np.linalg.norm(weights[i] - weights[j])
            idx[j] = j
        idx = np.argsort(x)
        x = x[idx]
        neighborhood[i][0:neighborhood_size] = idx[0:neighborhood_size]

    return neighborhood


def init_uniform_weights(population_size, nr_of_objectives):
    """Initializes uniform weights for the population.

    Args:
        population_size: Size of the population.
        nr_of_objectives: Number of objectives.

    Returns:
        A list of uniform weights.
    """
    # weights = np.random.dirichlet(np.ones(nr_of_objectives), size=population_size)
    if nr_of_objectives == 2:
        weights = das_dennis(99, 2)
    elif nr_of_objectives == 3:
        weights = das_dennis(13, 3)
    elif nr_of_objectives == 4:
        weights = das_dennis(7, 4)  # 120 preference vectors for 4 objectives
    return weights


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    pool = multiprocessing.Pool(1)

    # Create a specific folder for the objective function combination if specified
    objectives_subfolder = ""
    if "objective_names" in parameters and parameters["objective_names"]:
        objectives_subfolder = "/" + "_".join(parameters["objective_names"])

    folder = (
        DEFAULT_RESULTS_ROOT
        + "/MOEAD"  # Add algorithm-specific folder
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
        metavar="--file",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    print(f"Using config file: {args.config_file}")
    main(param_file=args.config_file)
