import os

if os.name == "posix":  # POSIX-compliant systems (Linux/Mac)
    BASE_PATH = "/evolutionary_algorithms/"
elif os.name == "nt":  # Windows systems
    BASE_PATH = "/evolutionary_algorithms/"
else:
    raise OSError(
        "Unsupported operating system. Please configure the BASE_PATH manually."
    )

DEFAULT_RESULTS_ROOT = "./results/single_runs"
REFERENCE_POINTS_FILE = BASE_PATH + "data/reference_points.json"
