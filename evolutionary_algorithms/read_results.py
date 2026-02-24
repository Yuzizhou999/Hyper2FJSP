"""Aggregate GA and MOEA/D result CSVs into simple summary stats."""

import csv
import os

csv.field_size_limit(100000000)
# Base directory for results
results_dir = "results/single_runs"


def calculate_averages():
    """Traverse GA/MOEAD run folders, aggregate hypervolume/runtime/HOF counts."""
    datasets = ["GA", "MOEAD"]
    metrics = {}

    for method in datasets:
        method_path = os.path.join(results_dir, method)
        for data_source in os.listdir(method_path):
            data_source_path = os.path.join(method_path, data_source)
            if not os.path.isdir(data_source_path):
                continue

            for dataset in os.listdir(data_source_path):
                dataset_path = os.path.join(data_source_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                hypervolumes_dict = {}
                runtimes_dict = {}
                unique_numbers_hof_dict = {}

                for root, _, files in os.walk(dataset_path):
                    # Only keep result folders that match the expected hyperparameter signature per method
                    if (
                        "ngen80000_pop100_cr0.7_indpb0.02" not in root
                        and method == "MOEAD"
                    ) or (
                        "ngen1000_pop100_cr0.7_indpb0.02" not in root and method == "GA"
                    ):
                        continue

                    # Extract objectives from the path
                    path_parts = root.split(os.sep)
                    if len(path_parts) < 6 or not path_parts[-2]:
                        continue

                    # Folder name one level above the CSV holds the objective tuple label (e.g., "mak_cos")
                    objectives = path_parts[-2]

                    if objectives not in hypervolumes_dict:
                        hypervolumes_dict[objectives] = []
                        runtimes_dict[objectives] = []
                        unique_numbers_hof_dict[objectives] = []

                    for file in files:
                        if file == "rseed5_results.csv":
                            file_path = os.path.join(root, file)

                            with open(file_path, "r") as csvfile:
                                reader = csv.DictReader(csvfile)
                                unique_hof = set()
                                for row in reader:
                                    try:
                                        hypervolumes_dict[objectives].append(
                                            float(row["hypervolume"])
                                        )
                                        runtimes_dict[objectives].append(
                                            float(row["runtime_seconds"])
                                        )
                                    except ValueError:
                                        pass

                                    # Parse and add unique tuples from hof
                                    # hof column stores a list of tuples as a string; eval to rebuild and count uniques
                                    hof = eval(row["hof"])
                                    unique_hof.update(hof)
                                    unique_numbers_hof_dict[objectives].append(
                                        len(unique_hof)
                                    )
                # Calculate averages for the dataset and objectives
                dataset_key = f"{method}/{data_source}/{dataset}"
                metrics[dataset_key] = {}
                for objectives in hypervolumes_dict:
                    avg_hypervolume = (
                        sum(hypervolumes_dict[objectives])
                        / len(hypervolumes_dict[objectives])
                        if hypervolumes_dict[objectives]
                        else 0
                    )
                    avg_runtime = (
                        sum(runtimes_dict[objectives]) / len(runtimes_dict[objectives])
                        if runtimes_dict[objectives]
                        else 0
                    )
                    unique_hof_count = (
                        sum(unique_numbers_hof_dict[objectives])
                        / len(unique_numbers_hof_dict[objectives])
                        if unique_numbers_hof_dict[objectives]
                        else 0
                    )
                    num_instances = len(hypervolumes_dict[objectives])

                    metrics[dataset_key][objectives] = {
                        "average_hypervolume": avg_hypervolume,
                        "average_runtime": avg_runtime,
                        "unique_hof_count": unique_hof_count,
                        "num_instances": num_instances,
                    }
    return metrics


if __name__ == "__main__":
    averages = calculate_averages()

    for dataset_objective, objectives_data in averages.items():
        print(f"Dataset: {dataset_objective}")
        for objectives, values in objectives_data.items():
            print(f"  Objectives: {objectives}")
            print(f"    Average Hypervolume: {values['average_hypervolume']}")
            print(f"    Average Runtime: {values['average_runtime']} seconds")
            print(f"    Unique HOF Count: {values['unique_hof_count']}")
            print(f"    Number of Instances: {values['num_instances']}")
