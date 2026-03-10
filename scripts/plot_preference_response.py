from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot preference-response curves from saved preference_responses."
    )
    parser.add_argument("--inst-dir", type=Path, required=True)
    parser.add_argument("--pref-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_response(directory: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(directory.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(f"No response files found in {directory}")

    preference_grid = None
    objective_stack = []
    objective_names = None

    for file_path in files:
        data = np.load(file_path, allow_pickle=True)
        preferences = np.asarray(data["preferences"], dtype=float)
        objectives = np.asarray(data["objectives"], dtype=float)
        names = [str(name) for name in data["objective_names"]]

        rounded_preferences = np.round(preferences, decimals=8)
        unique_preferences, inverse = np.unique(
            rounded_preferences, axis=0, return_inverse=True
        )
        aggregated_objectives = np.vstack(
            [
                objectives[inverse == idx].mean(axis=0)
                for idx in range(len(unique_preferences))
            ]
        )

        sort_idx = np.argsort(unique_preferences[:, 0])
        unique_preferences = unique_preferences[sort_idx]
        aggregated_objectives = aggregated_objectives[sort_idx]

        if preference_grid is None:
            preference_grid = unique_preferences
            objective_names = names
        else:
            if not np.allclose(preference_grid, unique_preferences):
                raise ValueError(
                    f"Preference grids do not match across files in {directory}"
                )

        objective_stack.append(aggregated_objectives)

    return preference_grid, np.stack(objective_stack).mean(axis=0), objective_names


def main() -> None:
    args = parse_args()

    inst_preferences, inst_objectives, objective_names = aggregate_response(args.inst_dir)
    pref_preferences, pref_objectives, pref_objective_names = aggregate_response(
        args.pref_dir
    )

    if objective_names != pref_objective_names:
        raise ValueError("Objective names differ between the two response directories.")
    if inst_objectives.shape[1] != 2:
        raise ValueError("This plotting script currently supports only 2 objectives.")

    x_inst = inst_preferences[:, 0]
    x_pref = pref_preferences[:, 0]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#fffdf8",
            "savefig.facecolor": "#f7f5ef",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    fig.suptitle(
        "Preference-Response Comparison",
        fontsize=16,
        fontweight="bold",
    )

    axes[0].plot(
        x_inst,
        inst_objectives[:, 0],
        color="#c46a1c",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Instance-aware",
    )
    axes[0].plot(
        x_pref,
        pref_objectives[:, 0],
        color="#1d6f8c",
        linewidth=2.2,
        marker="s",
        markersize=3,
        label="Preference-only",
    )
    axes[0].set_title(f"Mean {objective_names[0]} vs preference[0]")
    axes[0].set_ylabel(objective_names[0])
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(
        x_inst,
        inst_objectives[:, 1],
        color="#c46a1c",
        linewidth=2.2,
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        x_pref,
        pref_objectives[:, 1],
        color="#1d6f8c",
        linewidth=2.2,
        marker="s",
        markersize=3,
    )
    axes[1].set_title(f"Mean {objective_names[1]} vs preference[0]")
    axes[1].set_ylabel(objective_names[1])
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    axes[2].plot(
        inst_objectives[:, 0],
        inst_objectives[:, 1],
        color="#c46a1c",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Instance-aware",
    )
    axes[2].plot(
        pref_objectives[:, 0],
        pref_objectives[:, 1],
        color="#1d6f8c",
        linewidth=2.2,
        marker="s",
        markersize=3,
        label="Preference-only",
    )
    axes[2].set_title("Mean trade-off curve")
    axes[2].set_xlabel(objective_names[0])
    axes[2].set_ylabel(objective_names[1])
    axes[2].grid(alpha=0.25, linestyle="--", linewidth=0.8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
