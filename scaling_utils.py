from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalingBounds:
    """Common Q / temperature bounds shared across one comparison."""

    q_min: float
    q_max: float
    temp_min: float
    temp_max: float

    @property
    def q_span(self) -> float:
        return max(self.q_max - self.q_min, 1e-12)

    @property
    def temp_span(self) -> float:
        return max(self.temp_max - self.temp_min, 1e-12)


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected points with shape (n, 2), got {arr.shape}.")
    return arr


def compute_global_scaling_bounds(point_sets: list[np.ndarray]) -> ScalingBounds:
    """Build one common scaling from all scenarios in the current comparison."""

    if not point_sets:
        raise ValueError("At least one point set is required for global scaling.")

    arrays = [_as_points(points) for points in point_sets if len(points) > 0]
    if not arrays:
        raise ValueError("Point sets are empty; cannot compute scaling bounds.")

    merged = np.vstack(arrays)
    return ScalingBounds(
        q_min=float(np.min(merged[:, 0])),
        q_max=float(np.max(merged[:, 0])),
        temp_min=float(np.min(merged[:, 1])),
        temp_max=float(np.max(merged[:, 1])),
    )


def normalize_points(points: np.ndarray, bounds: ScalingBounds) -> np.ndarray:
    arr = _as_points(points)
    scaled = np.empty_like(arr, dtype=float)
    scaled[:, 0] = (arr[:, 0] - bounds.q_min) / bounds.q_span
    scaled[:, 1] = (arr[:, 1] - bounds.temp_min) / bounds.temp_span
    return scaled


def denormalize_points(points: np.ndarray, bounds: ScalingBounds) -> np.ndarray:
    arr = _as_points(points)
    raw = np.empty_like(arr, dtype=float)
    raw[:, 0] = bounds.q_min + arr[:, 0] * bounds.q_span
    raw[:, 1] = bounds.temp_min + arr[:, 1] * bounds.temp_span
    return raw
