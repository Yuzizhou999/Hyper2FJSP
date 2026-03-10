from __future__ import annotations

import numpy as np

try:
    import hvwfg  # type: ignore
except ModuleNotFoundError:
    hvwfg = None


def _pareto_filter(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    is_efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not is_efficient[i]:
            continue
        dominates = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominates[i] = False
        is_efficient[dominates] = False
    return points[is_efficient]


def _hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    valid_points = points[np.all(points < reference_point, axis=1)]
    if len(valid_points) == 0:
        return 0.0

    valid_points = np.unique(valid_points, axis=0)
    valid_points = _pareto_filter(valid_points)
    valid_points = valid_points[np.argsort(valid_points[:, 0])]

    hypervolume = 0.0
    current_y = reference_point[1]
    for x_value, y_value in valid_points:
        if y_value >= current_y:
            continue
        hypervolume += (reference_point[0] - x_value) * (current_y - y_value)
        current_y = y_value
    return float(hypervolume)


def compute_hypervolume(points, reference_point) -> float:
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if reference_point.ndim != 1:
        raise ValueError("reference_point must be a 1D array")
    if points.shape[1] != reference_point.shape[0]:
        raise ValueError("points and reference_point dimensions do not match")

    if hvwfg is not None:
        return float(hvwfg.wfg(points, reference_point))

    if points.shape[1] == 1:
        valid_points = points[points[:, 0] < reference_point[0]]
        if len(valid_points) == 0:
            return 0.0
        return float(reference_point[0] - np.min(valid_points[:, 0]))

    if points.shape[1] == 2:
        return _hypervolume_2d(points, reference_point)

    raise ModuleNotFoundError(
        "hvwfg is not installed, and the local fallback currently supports only 1D/2D hypervolume."
    )
