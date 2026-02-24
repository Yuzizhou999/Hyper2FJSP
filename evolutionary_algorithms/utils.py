import json
import os
import random
import shutil
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import toml
import torch
from genetic_algorithm.operators import ObjectiveFn
from pymoo.indicators.hv import Hypervolume

REFERENCE_POINT_INDICES = {
    ObjectiveFn.MAKESPAN: 0,
    ObjectiveFn.AVERAGE_FLOWTIME: 2,
    ObjectiveFn.TOTAL_TARDINESS: 7,
    ObjectiveFn.TOTAL_EARLINESS: 10,  # Default to alpha=0.5 (index 10)
    ObjectiveFn.COSTS: 9,
}


def get_earliness_reference_index(deadline_alpha):
    """
    Map deadline_alpha to the appropriate earliness reference point index.
    Index 10: alpha=0.5 (loose deadlines)
    Index 11: alpha=0.9 (medium deadlines)
    Index 12: alpha=1.0 (tight deadlines)
    """
    if deadline_alpha <= 0.51:
        return 10  # alpha 0.5
    elif deadline_alpha <= 0.91:
        return 11  # alpha 0.9
    else:
        return 12  # alpha 1.0


def load_config(config_filepath):
    with open(config_filepath, "r") as toml_file:
        return toml.load(toml_file)


def load_parameters(config_json):
    """Load parameters from a json file"""
    with open(config_json, "rb") as f:
        config_params = json.load(f)
    return config_params


def setup_device(config):
    device_type = config["general"]["device"]
    if device_type == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_type)
    print(
        f"Device set to: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}"
    )
    return device


def create_log_directory(config, config_file):
    model_logdir = (
        "trained_models/" + f"{config['ppo']['training_comment']}/"
        f"{config['ppo']['problem_type']}_"
        f"{config['environment']['problem_size']}_"
        f"{config['environment']['nr_objectives']}obj/"
        f"{datetime.now().strftime('%b%d_%H-%M-%S')}_"
        f"{socket.gethostname()}_r{random.randint(0, 100000)}"
    )

    os.makedirs(model_logdir, exist_ok=True)
    os.makedirs(model_logdir + "/intermediate_models/", exist_ok=True)
    os.makedirs(model_logdir + "/best_model/", exist_ok=True)
    shutil.copyfile(config_file, os.path.join(model_logdir, "train_config.toml"))
    print("started training with directory:", model_logdir)
    return model_logdir


def compute_hypervolume(population, nr_objectives, reference_point, lower_bounds=None):
    """
    :param population:
    :return: compute the hypervolume of a population
    """
    if hasattr(population[0], "fitness"):
        data = []
        objectives = [f"OBJ{i + 1}" for i in range(nr_objectives)]
        for individual in population:
            data_dict = {
                obj: individual.fitness.values[count]
                for count, obj in enumerate(objectives)
            }
            data.append(data_dict)

        df = pd.DataFrame(data)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Create the pointset
        pointset = np.array(
            [[df[obj].iloc[i] for obj in objectives] for i in range(len(df))]
        )

    else:
        # used in case of an ideal point (which is not a population, but just a point).
        pointset = np.array(population)

    # Compute hypervolume
    ref_point = np.array(reference_point)
    ref_point = (
        ref_point - np.array(lower_bounds) + 1e-8
        if lower_bounds is not None
        else ref_point
    )
    hv = Hypervolume(ref_point=ref_point)
    pointset = (
        pointset - np.array(lower_bounds) if lower_bounds is not None else pointset
    )
    hypervolume = hv.do(pointset)
    hypervolume = hypervolume / np.prod(ref_point)  # Normalize the hypervolume

    return round(hypervolume, 5)


def repair(ind):
    for i in range(len(ind)):
        if ind[i] < 0.0:
            ind[i] = 0.0
        elif ind[i] > 1.0:
            ind[i] = 1.0
    return ind


def create_stats_list(population, gen):
    stats_list = []
    for ind in population:
        tmp_dict = {}
        tmp_dict.update({"Generation": gen, "obj1": ind.fitness.values[0]})
        if hasattr(ind, "objectives"):
            tmp_dict.update(
                {
                    "obj1": ind.objectives[0],
                }
            )
        tmp_dict = {**tmp_dict}
        stats_list.append(tmp_dict)
    return stats_list


def record_stats(gen, population, logbook, stats, verbose, df_list, logging):
    stats_list = create_stats_list(population, gen)
    df_list.append(pd.DataFrame(stats_list))
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, **record)
    if verbose:
        logging.info(logbook.stream)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(
                ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1
            )


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)
