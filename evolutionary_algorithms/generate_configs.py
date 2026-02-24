import json
import os

# Base directories
data_dir = "data"
configs_dir = "configs"


# Template for the genetic algorithm configuration
def create_config(problem_instance, objective_names):
    return {
        "problem_instance": problem_instance,
        "nr_of_objectives": len(objective_names),
        "objective_names": objective_names,
        "population_size": 100,
        "ngen": 1000,
        "rseed": 5,
        "indpb": 0.02,
        "cr": 0.7,
        "logbook": True,
        "plotting": False,
    }


# Function to create configuration files
def generate_configs():
    objective_options = [
        ["makespan", "costs"],
        ["total_tardiness", "costs"],
        ["makespan", "average_flowtime", "costs"],
    ]

    for root, dirs, files in os.walk(data_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = create_config(relative_path, objective_names)

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_genetic_algorithm_configs(ngen=1000, subfolder="genetic_algorithm"):
    objective_options = [
        ["makespan", "costs"],
        ["total_tardiness", "costs"],
        ["makespan", "average_flowtime", "costs"],
    ]

    for root, dirs, files in os.walk(data_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = create_config(relative_path, objective_names)
                        config["ngen"] = ngen

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_moead_configs(ngen=80000, subfolder="moead"):
    objective_options = [
        ["makespan", "costs"],
        ["total_tardiness", "costs"],
        ["makespan", "average_flowtime", "costs"],
    ]

    for root, dirs, files in os.walk(data_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = {
                            "problem_instance": relative_path,
                            "nr_of_objectives": len(objective_names),
                            "objective_names": objective_names,
                            "population_size": 100,
                            "ngen": ngen,
                            "rseed": 5,
                            "indpb": 0.02,
                            "cr": 0.7,
                            "logbook": True,
                            "plotting": False,
                        }

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_ffsp_genetic_algorithm_configs(ngen=1000, subfolder="genetic_algorithm"):
    objective_options = [["makespan", "average_flowtime", "costs"]]

    # Only process FFSP data
    ffsp_dir = os.path.join(data_dir, "FFSP")
    for root, dirs, files in os.walk(ffsp_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = create_config(relative_path, objective_names)
                        config["ngen"] = ngen

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_ffsp_moead_configs(ngen=80000, subfolder="moead"):
    objective_options = [["makespan", "average_flowtime", "costs"]]

    # Only process FFSP data
    ffsp_dir = os.path.join(data_dir, "FFSP")
    for root, dirs, files in os.walk(ffsp_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = {
                            "problem_instance": relative_path,
                            "nr_of_objectives": len(objective_names),
                            "objective_names": objective_names,
                            "population_size": 100,
                            "ngen": ngen,
                            "rseed": 5,
                            "indpb": 0.02,
                            "cr": 0.7,
                            "logbook": True,
                            "plotting": False,
                        }

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_benchdata_genetic_algorithm_configs(
    ngen=1000, subfolder="genetic_algorithm"
):
    objective_options = [["makespan", "total_tardiness", "average_flowtime"]]

    # Only process BenchData
    benchdata_dir = os.path.join(data_dir, "BenchData")
    for root, dirs, files in os.walk(benchdata_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = create_config(relative_path, objective_names)
                        config["ngen"] = ngen

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


def generate_benchdata_moead_configs(ngen=80000, subfolder="moead"):
    objective_options = [["makespan", "total_tardiness", "average_flowtime"]]

    # Only process BenchData
    benchdata_dir = os.path.join(data_dir, "BenchData")
    for root, dirs, files in os.walk(benchdata_dir):
        for dir_index, (dirpath, _, filenames) in enumerate(os.walk(root)):
            for objective_names in objective_options:
                count = 0
                for file in filenames:
                    if file.endswith(".fjs"):
                        # Relative path of the problem instance
                        relative_path = os.path.relpath(
                            os.path.join(root, file), data_dir
                        )

                        # Replace backward slashes with forward slashes in the relative path
                        relative_path = relative_path.replace("\\", "/")

                        # Ensure the problem instance starts with a backslash
                        if not relative_path.startswith("/"):
                            relative_path = "/" + relative_path

                        # Subfolder for the current objective names
                        objective_folder = "_".join(objective_names)

                        # Corresponding config file path with numbered filenames
                        config_path = os.path.join(
                            configs_dir,
                            subfolder,
                            f"ngen{ngen}",
                            objective_folder,
                            os.path.relpath(dirpath, data_dir),
                            f"{count}.json",
                        )

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)

                        # Create the configuration
                        config = {
                            "problem_instance": relative_path,
                            "nr_of_objectives": len(objective_names),
                            "objective_names": objective_names,
                            "population_size": 100,
                            "ngen": ngen,
                            "rseed": 5,
                            "indpb": 0.02,
                            "cr": 0.7,
                            "logbook": True,
                            "plotting": False,
                        }

                        # Write the configuration to a JSON file
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=4)

                        count += 1


if __name__ == "__main__":
    # Generate BenchData configs with 1000 generations for GA and 80000 for MOEAD
    generate_benchdata_genetic_algorithm_configs(
        ngen=1000, subfolder="genetic_algorithm"
    )
    generate_benchdata_moead_configs(ngen=80000, subfolder="moead")
