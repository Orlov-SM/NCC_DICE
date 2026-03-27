from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scenario_config as sc
from geometry_utils import (
    SHAPELY_AVAILABLE,
    build_polygon_from_border,
    geometry_area,
    iter_polygon_patches,
    relative_area_loss,
)
from scaling_utils import compute_global_scaling_bounds, normalize_points


OUTPUT_DIR = Path("plots_png") / "compensation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPAIR_INVALID_POLYGONS = False
RUN_DEMO = True
RUN_MU_MAX_ANALYSIS = True

BASELINE_MU_MAX = sc.DEFAULT_MU_MAX
COMPARE_MU_MAX_VALUES = list(sc.MU_MAX_RANGE)
DEMO_COMPARED_MU_MAX = 1.4
DEMO_T_DEV = 5


@dataclass(frozen=True)
class BorderScenario:
    scenario_name: str
    sweep_value: float | str
    t_dev: int
    raw_points: np.ndarray
    scaled_points: np.ndarray


def _load_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got {points.shape[1]}.")
    return points


def load_mu_max_border(mu_max: float, t_dev: int) -> np.ndarray:
    path = Path(sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX, mu_max, t_dev))
    if not path.exists():
        raise FileNotFoundError(f"Missing border file: {path}")
    return _load_points(path)


def load_scaled_mu_max_pair(base_mu_max: float, comp_mu_max: float, t_dev: int) -> tuple[BorderScenario, BorderScenario]:
    base_raw = load_mu_max_border(base_mu_max, t_dev)
    comp_raw = load_mu_max_border(comp_mu_max, t_dev)
    bounds = compute_global_scaling_bounds([base_raw, comp_raw])

    base = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=base_mu_max,
        t_dev=t_dev,
        raw_points=base_raw,
        scaled_points=normalize_points(base_raw, bounds),
    )
    comp = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=comp_mu_max,
        t_dev=t_dev,
        raw_points=comp_raw,
        scaled_points=normalize_points(comp_raw, bounds),
    )
    return base, comp


def _plot_border(ax, points: np.ndarray, label: str, color: str) -> None:
    closed = np.vstack([points, points[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5, label=label)


def plot_raw_vs_scaled_comparison(base: BorderScenario, comp: BorderScenario, *, scaled: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    base_points = base.scaled_points if scaled else base.raw_points
    comp_points = comp.scaled_points if scaled else comp.raw_points
    _plot_border(ax, base_points, f"baseline mu_max={base.sweep_value:.2f}", "tab:blue")
    _plot_border(ax, comp_points, f"compensated mu_max={comp.sweep_value:.2f}", "tab:orange")
    ax.grid(True)
    ax.legend(loc="best")

    if scaled:
        ax.set_xlabel("Scaled Q")
        ax.set_ylabel("Scaled Delta T")
        ax.set_title(f"Scaled comparison at t_dev={base.t_dev}")
        out_path = OUTPUT_DIR / f"scaled_mu{base.sweep_value:.2f}_vs_{comp.sweep_value:.2f}_tdev{base.t_dev}.png"
    else:
        ax.set_xlabel("Q in 2100")
        ax.set_ylabel("Delta T in 2100 (C)")
        ax.set_title(f"Raw comparison at t_dev={base.t_dev}")
        out_path = OUTPUT_DIR / f"raw_mu{base.sweep_value:.2f}_vs_{comp.sweep_value:.2f}_tdev{base.t_dev}.png"

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_lost_region(base: BorderScenario, comp: BorderScenario, *, annotate_loss: float) -> tuple[Path, dict[str, float]]:
    base_polygon, base_debug = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    comp_polygon, comp_debug = build_polygon_from_border(
        comp.scaled_points,
        label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    lost_area, relative_loss, lost_geometry = relative_area_loss(base_polygon, comp_polygon)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    _plot_border(ax, base.scaled_points, f"baseline mu_max={base.sweep_value:.2f}", "tab:blue")
    _plot_border(ax, comp.scaled_points, f"compensated mu_max={comp.sweep_value:.2f}", "tab:orange")

    for coords in iter_polygon_patches(lost_geometry):
        ax.fill(coords[:, 0], coords[:, 1], color="tab:red", alpha=0.25, label="Lost region")

    ax.set_xlabel("Scaled Q")
    ax.set_ylabel("Scaled Delta T")
    ax.set_title(f"Lost region at t_dev={base.t_dev}")
    ax.grid(True)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"Relative loss = {annotate_loss:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"loss_region_mu{base.sweep_value:.2f}_vs_{comp.sweep_value:.2f}_tdev{base.t_dev}.png"
    plt.savefig(out_path)
    plt.close(fig)

    debug_stats = {
        "base_area_scaled": geometry_area(base_polygon),
        "comp_area_scaled": geometry_area(comp_polygon),
        "lost_area_scaled": lost_area,
        "relative_loss": relative_loss,
        "base_signed_area": base_debug.signed_area,
        "comp_signed_area": comp_debug.signed_area,
    }
    return out_path, debug_stats


def compute_mu_max_loss_series(base_mu_max: float, compare_values: list[float]) -> dict[int, list[tuple[float, float]]]:
    baseline_t_devs = set(sc.t_dev_runs(base_mu_max, base_mu_max))
    losses_by_t_dev: dict[int, list[tuple[float, float]]] = {}

    for comp_mu_max in compare_values:
        common_t_devs = sorted(baseline_t_devs.intersection(sc.t_dev_runs(comp_mu_max, base_mu_max)))
        for t_dev in common_t_devs:
            base, comp = load_scaled_mu_max_pair(base_mu_max, comp_mu_max, t_dev)
            base_polygon, _ = build_polygon_from_border(
                base.scaled_points,
                label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            losses_by_t_dev.setdefault(t_dev, []).append((float(comp_mu_max), relative_loss))

    return losses_by_t_dev


def plot_mu_max_loss_series(losses_by_t_dev: dict[int, list[tuple[float, float]]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for t_dev, values in sorted(losses_by_t_dev.items()):
        values = sorted(values, key=lambda item: item[0])
        ax.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=1.5,
            label=f"t_dev={t_dev}",
        )

    ax.set_xlabel("mu_max")
    ax.set_ylabel("Relative loss")
    ax.set_title("Scaled area loss by mu_max")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "mu_max_relative_loss.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def run_demo() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("Demo requires shapely.")

    base, comp = load_scaled_mu_max_pair(BASELINE_MU_MAX, DEMO_COMPARED_MU_MAX, DEMO_T_DEV)
    raw_path = plot_raw_vs_scaled_comparison(base, comp, scaled=False)
    scaled_path = plot_raw_vs_scaled_comparison(base, comp, scaled=True)
    lost_region_path, debug_stats = plot_lost_region(base, comp, annotate_loss=0.0)

    # Recompute once for logging and validation. The self-comparison should be ~0.
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline self-check mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    comp_polygon, _ = build_polygon_from_border(
        comp.scaled_points,
        label=f"comp-check mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
    _, self_relative_loss, _ = relative_area_loss(base_polygon, base_polygon)

    larger_polygon = base_polygon if geometry_area(base_polygon) >= geometry_area(comp_polygon) else comp_polygon
    smaller_polygon = comp_polygon if larger_polygon is base_polygon else base_polygon
    _, positive_check_loss, _ = relative_area_loss(larger_polygon, smaller_polygon)

    print(f"Raw comparison plot: {raw_path}")
    print(f"Scaled comparison plot: {scaled_path}")
    print(f"Lost-region plot: {lost_region_path}")
    print(
        "Scaled areas: "
        f"base={debug_stats['base_area_scaled']:.6f}, "
        f"comp={debug_stats['comp_area_scaled']:.6f}, "
        f"lost={debug_stats['lost_area_scaled']:.6f}"
    )
    print(f"Relative loss(base, comp) = {relative_loss:.6f}")
    print(f"Self-comparison loss(base, base) = {self_relative_loss:.6e}")
    print(f"Positive-loss debug check(larger, smaller) = {positive_check_loss:.6f}")

    # Update the annotation after computing the true value.
    final_lost_region_path, _ = plot_lost_region(base, comp, annotate_loss=relative_loss)
    if final_lost_region_path != lost_region_path:
        print(f"Updated lost-region plot: {final_lost_region_path}")


def run_mu_max_analysis() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("mu_max analysis requires shapely.")
    losses_by_t_dev = compute_mu_max_loss_series(BASELINE_MU_MAX, COMPARE_MU_MAX_VALUES)
    out_path = plot_mu_max_loss_series(losses_by_t_dev)
    print(f"Saved mu_max loss plot: {out_path}")
    for t_dev, values in sorted(losses_by_t_dev.items()):
        print(f"t_dev={t_dev}: {values}")


if __name__ == "__main__":
    if RUN_DEMO:
        run_demo()
    if RUN_MU_MAX_ANALYSIS:
        run_mu_max_analysis()
