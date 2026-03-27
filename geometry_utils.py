from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.validation import explain_validity

    SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - guarded at runtime in this repo
    make_valid = None
    GeometryCollection = None
    MultiPolygon = None
    Polygon = None
    explain_validity = None
    SHAPELY_AVAILABLE = False


@dataclass(frozen=True)
class PolygonDebugInfo:
    label: str
    point_count: int
    signed_area: float
    is_closed_in_input: bool
    is_valid: bool
    validity_message: str
    used_repair: bool


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected points with shape (n, 2), got {arr.shape}.")
    if len(arr) < 3:
        raise ValueError("At least 3 border points are required to build a polygon.")
    return arr


def _strip_duplicate_closure(points: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = _as_points(points)
    is_closed = bool(np.allclose(arr[0], arr[-1]))
    if is_closed:
        arr = arr[:-1]
    if len(arr) < 3:
        raise ValueError("Polygon border is degenerate after removing duplicate closure.")
    return arr, is_closed


def signed_area(points: np.ndarray) -> float:
    arr, _ = _strip_duplicate_closure(points)
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_area_from_points(points: np.ndarray) -> float:
    return abs(signed_area(points))


def require_shapely() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError(
            "Shapely is required for polygon validity, intersection, and difference operations. "
            "Install it in the project virtual environment to enable compensation analysis."
        )


def build_polygon_from_border(
    points: np.ndarray,
    *,
    label: str = "",
    repair_invalid: bool = False,
) -> tuple[Polygon | MultiPolygon, PolygonDebugInfo]:
    """
    Build a polygon from the existing border ordering only.

    The code intentionally does not reorder points and does not replace the
    border by a convex hull. If the provided border is invalid, the caller gets
    an explicit validity report and may opt into repair_invalid.
    """

    require_shapely()
    arr, is_closed = _strip_duplicate_closure(points)
    base_polygon = Polygon(arr)
    validity_message = explain_validity(base_polygon)
    is_valid = bool(base_polygon.is_valid and not base_polygon.is_empty)
    used_repair = False
    polygon = base_polygon

    if not is_valid:
        if not repair_invalid:
            debug = PolygonDebugInfo(
                label=label,
                point_count=len(arr),
                signed_area=signed_area(arr),
                is_closed_in_input=is_closed,
                is_valid=False,
                validity_message=validity_message,
                used_repair=False,
            )
            raise ValueError(f"Invalid polygon for {label or 'unnamed border'}: {validity_message}")

        polygon = make_valid(base_polygon)
        used_repair = True
        validity_message = explain_validity(polygon)
        is_valid = bool(polygon.is_valid and not polygon.is_empty)
        if not is_valid:
            raise ValueError(
                f"Polygon for {label or 'unnamed border'} remained invalid after repair: {validity_message}"
            )

    debug = PolygonDebugInfo(
        label=label,
        point_count=len(arr),
        signed_area=signed_area(arr),
        is_closed_in_input=is_closed,
        is_valid=is_valid,
        validity_message=validity_message,
        used_repair=used_repair,
    )
    return polygon, debug


def geometry_area(geometry) -> float:
    if geometry is None:
        return 0.0
    return float(geometry.area)


def geometry_intersection(base_geometry, other_geometry):
    require_shapely()
    return base_geometry.intersection(other_geometry)


def geometry_difference(base_geometry, other_geometry):
    require_shapely()
    return base_geometry.difference(other_geometry)


def iter_polygon_patches(geometry):
    """Yield exterior coordinates for Polygon / MultiPolygon geometries."""

    require_shapely()
    if geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield np.asarray(geometry.exterior.coords)
        return

    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield np.asarray(polygon.exterior.coords)
        return

    if isinstance(geometry, GeometryCollection):
        for sub_geometry in geometry.geoms:
            if isinstance(sub_geometry, Polygon):
                yield np.asarray(sub_geometry.exterior.coords)
            elif isinstance(sub_geometry, MultiPolygon):
                for polygon in sub_geometry.geoms:
                    yield np.asarray(polygon.exterior.coords)


def relative_area_loss(base_geometry, compensated_geometry) -> tuple[float, float, object]:
    lost_geometry = geometry_difference(base_geometry, compensated_geometry)
    base_area = geometry_area(base_geometry)
    lost_area = geometry_area(lost_geometry)
    relative_loss = 0.0 if base_area <= 0.0 else lost_area / base_area
    return lost_area, relative_loss, lost_geometry
