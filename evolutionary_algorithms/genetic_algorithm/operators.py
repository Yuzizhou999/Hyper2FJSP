import random
from enum import Enum

import numpy as np
from helper_functions import update_operations_available_for_scheduling
from heuristics_scheduler.heuristics import (
    global_selection_scheduler,
    local_selection_scheduler,
    random_scheduler,
)
from scheduling_environment.jobShop import JobShop
from scheduling_environment.operation import Operation


class ObjectiveFn(Enum):
    MAKESPAN = "makespan"
    TOTAL_TARDINESS = "total_tardiness"
    TOTAL_EARLINESS = "total_earliness"
    AVERAGE_FLOWTIME = "average_flowtime"
    NEGATIVE_MAKESPAN = "negative_makespan"
    COSTS = "costs"


def select_next_operation_from_job(jobShopEnv: JobShop, job_id) -> Operation:
    # select next operation for job
    for operation in jobShopEnv.operations_available_for_scheduling:
        if operation.job_id == job_id:
            return operation


def pox_crossover(ind1, ind2, nr_preserving_jobs):
    preserving_jobs = random.sample(range(1, max(ind1)), nr_preserving_jobs)

    new_sequence_ind1 = list(filter(lambda a: a not in preserving_jobs, ind2))
    for i in range(len(ind1)):
        if ind1[i] in preserving_jobs:
            new_sequence_ind1.insert(i, ind1[i])

    new_sequence_ind2 = list(filter(lambda a: a not in preserving_jobs, ind1))
    for i in range(len(ind2)):
        if ind2[i] in preserving_jobs:
            new_sequence_ind2.insert(i, ind2[i])

    return new_sequence_ind1, new_sequence_ind2


def mutate_shortest_proc_time(individual, indpb, jobShopEnv: JobShop):
    for i, _ in enumerate(individual):
        if random.random() < indpb:
            operation = jobShopEnv.operations[i]
            individual[i] = np.argmin(operation.processing_times)
    return individual


def mutate_sequence_exchange(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            j = random.choice([index for index in range(len(individual)) if index != i])
            individual[i], individual[j] = individual[j], individual[i]
    return individual


# Initialize an individual for the genetic algorithm (with random actions selection heuristic)
def init_individual(ind_class, params, jobShopEnv):
    """create individual, indivial is a list of machine selection (ix of options) and operation sequence (ix of job)"""

    rand = random.random()
    if rand <= 0.6:  # 60% initial assignment with global selection scheduler
        jobShopEnv = global_selection_scheduler(jobShopEnv)
    elif rand <= 0.9:  # 30% initial assignment with local selection scheduler
        jobShopEnv = local_selection_scheduler(jobShopEnv)
    else:  # 10% initial assignment with random scheduler
        jobShopEnv = random_scheduler(jobShopEnv)

    # get the operation sequence and machine allocation lists
    operation_sequence = [
        operation.job_id for operation in jobShopEnv.scheduled_operations
    ]
    machine_selection = [
        (
            operation.operation_id,
            sorted(list(operation.processing_times.keys())).index(
                operation.scheduled_machine
            ),
        )
        for operation in jobShopEnv.scheduled_operations
    ]
    machine_selection.sort()
    machine_selection = [allocation for _, allocation in machine_selection]
    jobShopEnv.reset()
    return ind_class([machine_selection, operation_sequence])


# Initialize a population
def init_population(toolbox, population_size):
    return [toolbox.init_individual() for _ in range(population_size)]


def evaluate_individual(
    individual,
    jobShopEnv: JobShop,
    objectives,
    objective_names: list | None = None,
    reset=True,
    deadline_alpha: float = 0.5,
):
    jobShopEnv.reset()
    update_operations_available_for_scheduling(jobShopEnv)
    for i in range(len(individual[0])):
        job_id = individual[1][i]
        operation = select_next_operation_from_job(jobShopEnv, job_id)
        operation_option_index = individual[0][operation.operation_id]
        machine_id = sorted(operation.processing_times.keys())[operation_option_index]
        duration = operation.processing_times[machine_id]

        jobShopEnv.schedule_operation_with_backfilling(operation, machine_id, duration)
        update_operations_available_for_scheduling(jobShopEnv)
    if objective_names is not None:
        objective_names = [
            ObjectiveFn(str(obj_fn).lower()) for obj_fn in objective_names
        ]
        obj_values = []
        # Iterate through objective_names in the order they were specified
        for obj in objective_names:
            if obj == ObjectiveFn.MAKESPAN:
                obj_values.append(jobShopEnv.makespan)
            elif obj == ObjectiveFn.TOTAL_TARDINESS:
                obj_values.append(jobShopEnv.total_tardiness(deadline_alpha))
            elif obj == ObjectiveFn.TOTAL_EARLINESS:
                obj_values.append(jobShopEnv.total_earliness(deadline_alpha))
            elif obj == ObjectiveFn.AVERAGE_FLOWTIME:
                obj_values.append(jobShopEnv.average_flowtime)
            elif obj == ObjectiveFn.COSTS:
                obj_values.append(jobShopEnv.total_costs())
            elif obj == ObjectiveFn.NEGATIVE_MAKESPAN:
                obj_values.append(-jobShopEnv.makespan)
        if reset:
            jobShopEnv.reset()
        return tuple(obj_values), jobShopEnv

    elif objectives == 1:
        makespan = jobShopEnv.makespan
        if reset:
            jobShopEnv.reset()
        return (makespan), jobShopEnv

    elif objectives == 2:
        makespan = jobShopEnv.makespan
        # balanced_workload = jobShopEnv.balanced_workload
        # total_tardiness_05 = jobShopEnv.total_tardiness(0.5)
        total_costs = jobShopEnv.total_costs()
        if reset:
            jobShopEnv.reset()
        return (makespan, total_costs), jobShopEnv

    elif objectives == 3:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        if reset:
            jobShopEnv.reset()
        return (makespan, balanced_workload, average_flowtime), jobShopEnv

    elif objectives == 4:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        total_workload = jobShopEnv.total_workload
        if reset:
            jobShopEnv.reset()
        return (
            makespan,
            balanced_workload,
            average_flowtime,
            total_workload,
        ), jobShopEnv

    elif objectives == 5:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        total_workload = jobShopEnv.total_workload
        max_flowtime = jobShopEnv.max_flowtime
        if reset:
            jobShopEnv.reset()
        return (
            makespan,
            balanced_workload,
            average_flowtime,
            total_workload,
            max_flowtime,
        ), jobShopEnv

    elif objectives == 6:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        total_workload = jobShopEnv.total_workload
        max_flowtime = jobShopEnv.max_flowtime
        average_workload = jobShopEnv.average_workload
        if reset:
            jobShopEnv.reset()
        return (
            makespan,
            balanced_workload,
            average_flowtime,
            total_workload,
            max_flowtime,
            average_workload,
        ), jobShopEnv

    elif objectives == 7:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        total_workload = jobShopEnv.total_workload
        max_flowtime = jobShopEnv.max_flowtime
        average_workload = jobShopEnv.average_workload
        max_workload = jobShopEnv.max_workload
        if reset:
            jobShopEnv.reset()
        return (
            makespan,
            balanced_workload,
            average_flowtime,
            total_workload,
            max_flowtime,
            average_workload,
            max_workload,
        ), jobShopEnv

    elif objectives == 9:
        makespan = jobShopEnv.makespan
        balanced_workload = jobShopEnv.balanced_workload
        average_flowtime = jobShopEnv.average_flowtime
        total_workload = jobShopEnv.total_workload
        max_flowtime = jobShopEnv.max_flowtime
        average_workload = jobShopEnv.average_workload
        max_workload = jobShopEnv.max_workload
        total_tardiness_05 = jobShopEnv.total_tardiness(0.5)
        total_tardiness_1 = jobShopEnv.total_tardiness(1.0)
        if reset:
            jobShopEnv.reset()
        return (
            makespan,
            balanced_workload,
            average_flowtime,
            total_workload,
            max_flowtime,
            average_workload,
            max_workload,
            total_tardiness_05,
            total_tardiness_1,
        ), jobShopEnv


def evaluate_population(toolbox, population, objectives, logging):

    # parallel evaluation of population
    population = [[ind[0], ind[1]] for ind in population]
    fitnesses = toolbox.map(toolbox.evaluate_individual, population)
    fitnesses = [fit[0] for fit in fitnesses]

    return fitnesses


def variation(population, toolbox, lambda_, cr, indpb):
    offspring = []
    for _ in range(int(lambda_)):
        op_choice = random.random()
        if op_choice < cr:  # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            if random.random() < 0.5:
                ind1[0], ind2[0] = toolbox.mate_TwoPoint(ind1[0], ind2[0])
            else:
                ind1[0], ind2[0] = toolbox.mate_Uniform(ind1[0], ind2[0])

            ind1[1], ind2[1] = toolbox.mate_POX(ind1[1], ind2[1])
            del ind1.fitness.values, ind2.fitness.values

        else:  # Apply reproduction
            ind1 = toolbox.clone(random.choice(population))

        # Apply mutation
        ind1[0] = toolbox.mutate_machine_selection(ind1[0], indpb)
        ind1[1] = toolbox.mutate_operation_sequence(ind1[1], indpb)

        del ind1.fitness.values
        offspring.append(ind1)

    return offspring


def repair_precedence_constraints(env, offspring):
    precedence_relations = env.precedence_relations_jobs
    for ind in offspring:
        i = 0
        lst = ind[1]
        while i < len(ind[1]):
            if lst[i] in precedence_relations.keys():
                max_index = 0
                for j in precedence_relations[lst[i]]:
                    index = len(lst) - 1 - lst[::-1].index(j)
                    if index > max_index:
                        max_index = index
                if max_index > i:
                    item = lst[i]
                    lst.pop(i)  # Remove the item from the source index
                    lst.insert(max_index, item)
                    continue
            i += 1
    return offspring
