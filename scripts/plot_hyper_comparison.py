from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward and validation comparisons for HYPER ablations."
    )
    parser.add_argument("--inst-reward", type=Path, required=True)
    parser.add_argument("--pref-reward", type=Path, required=True)
    parser.add_argument("--inst-vali", type=Path, required=True)
    parser.add_argument("--pref-vali", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window size for reward smoothing.",
    )
    return parser.parse_args()


def load_reward_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    content = path.read_text(encoding="utf-8")
    data = eval(content, {"array": np.array})
    updates = np.array([entry[0] for entry in data], dtype=int)
    rewards = np.stack([entry[1] for entry in data]).astype(float)
    return updates, rewards


def load_validation_log(path: Path) -> np.ndarray:
    content = path.read_text(encoding="utf-8")
    logged_values = np.array(eval(content, {}), dtype=float)
    return -logged_values


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window > len(values):
        window = len(values)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def plot_reward_axis(
    ax: plt.Axes,
    updates: np.ndarray,
    inst_values: np.ndarray,
    pref_values: np.ndarray,
    title: str,
    ylabel: str,
    window: int,
) -> None:
    ax.plot(
        updates,
        inst_values,
        color="#c46a1c",
        alpha=0.18,
        linewidth=1.0,
    )
    ax.plot(
        updates,
        pref_values,
        color="#1d6f8c",
        alpha=0.18,
        linewidth=1.0,
    )
    ax.plot(
        updates,
        moving_average(inst_values, window),
        color="#c46a1c",
        linewidth=2.2,
        label="Instance-aware",
    )
    ax.plot(
        updates,
        moving_average(pref_values, window),
        color="#1d6f8c",
        linewidth=2.2,
        label="Preference-only",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)


def plot_validation_axis(
    ax: plt.Axes,
    inst_hv: np.ndarray,
    pref_hv: np.ndarray,
) -> None:
    validation_updates = np.arange(1, len(inst_hv) + 1) * 40
    ax.plot(
        validation_updates,
        inst_hv,
        color="#c46a1c",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Instance-aware",
    )
    ax.plot(
        validation_updates,
        pref_hv,
        color="#1d6f8c",
        linewidth=2.2,
        marker="s",
        markersize=4,
        label="Preference-only",
    )

    inst_best_idx = int(np.argmax(inst_hv))
    pref_best_idx = int(np.argmax(pref_hv))
    ax.scatter(
        validation_updates[inst_best_idx],
        inst_hv[inst_best_idx],
        color="#c46a1c",
        s=55,
        zorder=5,
    )
    ax.scatter(
        validation_updates[pref_best_idx],
        pref_hv[pref_best_idx],
        color="#1d6f8c",
        s=55,
        zorder=5,
    )
    ax.annotate(
        f"Best {inst_hv[inst_best_idx]:.3f}\n@ {validation_updates[inst_best_idx]}",
        (validation_updates[inst_best_idx], inst_hv[inst_best_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        color="#8d4d12",
        fontsize=9,
    )
    ax.annotate(
        f"Best {pref_hv[pref_best_idx]:.3f}\n@ {validation_updates[pref_best_idx]}",
        (validation_updates[pref_best_idx], pref_hv[pref_best_idx]),
        xytext=(8, -28),
        textcoords="offset points",
        color="#144b60",
        fontsize=9,
    )
    ax.set_title("Validation Hypervolume", fontsize=12, pad=10)
    ax.set_xlabel("Update")
    ax.set_ylabel("Normalized HV")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)


def main() -> None:
    args = parse_args()

    inst_updates, inst_rewards = load_reward_log(args.inst_reward)
    pref_updates, pref_rewards = load_reward_log(args.pref_reward)
    inst_hv = load_validation_log(args.inst_vali)
    pref_hv = load_validation_log(args.pref_vali)

    if not np.array_equal(inst_updates, pref_updates):
        raise ValueError("Reward logs do not share the same update indices.")
    if len(inst_hv) != len(pref_hv):
        raise ValueError("Validation logs do not share the same number of checkpoints.")

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f8f6f1",
            "axes.facecolor": "#fffdf8",
            "savefig.facecolor": "#f8f6f1",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    fig.suptitle(
        "HYPER Ablation on SD1 10x5 (Makespan + Costs)",
        fontsize=16,
        fontweight="bold",
    )

    plot_reward_axis(
        axes[0],
        inst_updates,
        inst_rewards[:, 0],
        pref_rewards[:, 0],
        title="Reward Trend: Objective 1",
        ylabel="Reward",
        window=args.window,
    )
    plot_reward_axis(
        axes[1],
        inst_updates,
        inst_rewards[:, 1],
        pref_rewards[:, 1],
        title="Reward Trend: Objective 2",
        ylabel="Reward",
        window=args.window,
    )
    plot_validation_axis(axes[2], inst_hv, pref_hv)

    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
