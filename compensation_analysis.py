from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scenario_config as sc
from geometry_utils import (
    SHAPELY_AVAILABLE,
    build_polygon_from_border,
    geometry_area,
    iter_polygon_patches,
    relative_area_loss,
)
from scaling_utils import ScalingBounds, normalize_points


OUTPUT_DIR = Path("plots_png") / "compensation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.8,
    }
)

REPAIR_INVALID_POLYGONS = False
RUN_DEMO = True
RUN_MU_MAX_ANALYSIS = True
RUN_MU_12_TDEV_SERIES = True
RUN_MU_MAX_LOSS_REGIONS = False
RUN_DELAY_COMPENSATION_EXAMPLE = True
RUN_DELAY_COMPENSATION_SERIES = True

BASELINE_MU_MAX = sc.DEFAULT_MU_MAX
SCALING_BASELINE_MU_MAX = 1.2
SCALING_BASELINE_T_DEV = 1
COMPARE_MU_MAX_VALUES = [round(value, 2) for value in np.arange(1.2, 2.01, 0.1)]
TARGET_T_DEVS = [4, 5, 6, 7, 8, 9]
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


def t_dev_to_year(t_dev: int) -> int:
    return 2015 + 5 * int(t_dev)


def _scenario_role_label(base: BorderScenario, comp: BorderScenario) -> str:
    if np.isclose(float(comp.sweep_value), float(base.sweep_value)) and comp.t_dev != base.t_dev:
        return f"Delayed mitigation to {t_dev_to_year(comp.t_dev)}"
    if float(comp.sweep_value) > float(base.sweep_value):
        return f"Compensated to mu_max={float(comp.sweep_value):.2f}, year {t_dev_to_year(comp.t_dev)}"
    return f"Comparison year {t_dev_to_year(comp.t_dev)}"


def load_mu_max_border(mu_max: float, t_dev: int) -> np.ndarray:
    path = Path(sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX, mu_max, t_dev))
    if not path.exists():
        raise FileNotFoundError(f"Missing border file: {path}")
    return _load_points(path)


def load_fixed_scaling_bounds() -> ScalingBounds:
    """
    Use one fixed scaling source for every compensation-analysis comparison.

    This keeps all computed areas in the same scaled coordinate system:
    scaling is derived only from the baseline border mu=1.2, t_dev=1.
    """

    baseline_raw = load_mu_max_border(SCALING_BASELINE_MU_MAX, SCALING_BASELINE_T_DEV)
    return ScalingBounds(
        q_min=float(np.min(baseline_raw[:, 0])),
        q_max=float(np.max(baseline_raw[:, 0])),
        temp_min=float(np.min(baseline_raw[:, 1])),
        temp_max=float(np.max(baseline_raw[:, 1])),
    )


def available_t_devs_for_mu(mu_max: float) -> list[int]:
    available = []
    for t_dev in TARGET_T_DEVS + [SCALING_BASELINE_T_DEV]:
        try:
            load_mu_max_border(mu_max, t_dev)
            available.append(t_dev)
        except FileNotFoundError:
            continue
    return sorted(set(available))


@lru_cache(maxsize=1)
def common_scaled_axis_limits() -> tuple[float, float, float, float]:
    bounds = load_fixed_scaling_bounds()
    all_points = []
    for mu_max in COMPARE_MU_MAX_VALUES:
        for t_dev in available_t_devs_for_mu(mu_max):
            try:
                raw_points = load_mu_max_border(mu_max, t_dev)
            except FileNotFoundError:
                continue
            all_points.append(normalize_points(raw_points, bounds))

    if not all_points:
        return (-0.02, 1.02, -0.02, 1.02)

    merged = np.vstack(all_points)
    x_min, x_max = float(np.min(merged[:, 0])), float(np.max(merged[:, 0]))
    y_min, y_max = float(np.min(merged[:, 1])), float(np.max(merged[:, 1]))
    x_pad = max(0.03, 0.05 * (x_max - x_min))
    y_pad = max(0.03, 0.05 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


@lru_cache(maxsize=1)
def common_raw_axis_limits() -> tuple[float, float, float, float]:
    all_points = []
    for mu_max in COMPARE_MU_MAX_VALUES:
        for t_dev in available_t_devs_for_mu(mu_max):
            try:
                all_points.append(load_mu_max_border(mu_max, t_dev))
            except FileNotFoundError:
                continue

    if not all_points:
        return (0.0, 1.0, 0.0, 1.0)

    merged = np.vstack(all_points)
    x_min, x_max = float(np.min(merged[:, 0])), float(np.max(merged[:, 0]))
    y_min, y_max = float(np.min(merged[:, 1])), float(np.max(merged[:, 1]))
    x_pad = max(10.0, 0.05 * (x_max - x_min))
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad, max(0.0, y_min - y_pad), y_max + y_pad)


def load_scaled_mu_max_pair(base_mu_max: float, comp_mu_max: float, t_dev: int) -> tuple[BorderScenario, BorderScenario]:
    base_raw = load_mu_max_border(base_mu_max, t_dev)
    comp_raw = load_mu_max_border(comp_mu_max, t_dev)
    bounds = load_fixed_scaling_bounds()

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


def load_scaled_mu_max_comparison(
    base_mu_max: float,
    base_t_dev: int,
    comp_mu_max: float,
    comp_t_dev: int,
) -> tuple[BorderScenario, BorderScenario]:
    bounds = load_fixed_scaling_bounds()
    base_raw = load_mu_max_border(base_mu_max, base_t_dev)
    comp_raw = load_mu_max_border(comp_mu_max, comp_t_dev)

    base = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=base_mu_max,
        t_dev=base_t_dev,
        raw_points=base_raw,
        scaled_points=normalize_points(base_raw, bounds),
    )
    comp = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=comp_mu_max,
        t_dev=comp_t_dev,
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
    _plot_border(
        ax,
        base_points,
        f"Baseline mu_max={base.sweep_value:.2f}, year {t_dev_to_year(base.t_dev)}",
        "tab:blue",
    )
    _plot_border(ax, comp_points, _scenario_role_label(base, comp), "tab:orange")
    ax.grid(True)
    ax.legend(loc="best")

    if scaled:
        ax.set_xlabel("Scaled Q")
        ax.set_ylabel("Scaled Delta T")
        ax.set_title(
            f"Scaled comparison: baseline {t_dev_to_year(base.t_dev)} vs comparison {t_dev_to_year(comp.t_dev)}"
        )
        ax.set_xlim(*common_scaled_axis_limits()[:2])
        ax.set_ylim(*common_scaled_axis_limits()[2:])
        out_path = OUTPUT_DIR / (
            f"scaled_mu{base.sweep_value:.2f}_tdev{base.t_dev}"
            f"_vs_mu{comp.sweep_value:.2f}_tdev{comp.t_dev}.png"
        )
    else:
        ax.set_xlabel("Q in 2100")
        ax.set_ylabel("Delta T in 2100 (C)")
        ax.set_title(
            f"Raw comparison: baseline {t_dev_to_year(base.t_dev)} vs comparison {t_dev_to_year(comp.t_dev)}"
        )
        ax.set_xlim(*common_raw_axis_limits()[:2])
        ax.set_ylim(*common_raw_axis_limits()[2:])
        out_path = OUTPUT_DIR / (
            f"raw_mu{base.sweep_value:.2f}_tdev{base.t_dev}"
            f"_vs_mu{comp.sweep_value:.2f}_tdev{comp.t_dev}.png"
        )

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
    _plot_border(
        ax,
        base.scaled_points,
        f"Baseline mu_max={base.sweep_value:.2f}, year {t_dev_to_year(base.t_dev)}",
        "tab:blue",
    )
    _plot_border(ax, comp.scaled_points, _scenario_role_label(base, comp), "tab:orange")

    lost_region_labeled = False
    for coords in iter_polygon_patches(lost_geometry):
        ax.fill(
            coords[:, 0],
            coords[:, 1],
            color="tab:red",
            alpha=0.25,
            label="Lost region" if not lost_region_labeled else None,
        )
        lost_region_labeled = True

    ax.set_xlabel("Scaled Q")
    ax.set_ylabel("Scaled Delta T")
    ax.set_title(
        f"Loss region: baseline {t_dev_to_year(base.t_dev)} vs comparison {t_dev_to_year(comp.t_dev)}"
    )
    ax.set_xlim(*common_scaled_axis_limits()[:2])
    ax.set_ylim(*common_scaled_axis_limits()[2:])
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

    out_path = OUTPUT_DIR / (
        f"loss_region_mu{base.sweep_value:.2f}_tdev{base.t_dev}"
        f"_vs_mu{comp.sweep_value:.2f}_tdev{comp.t_dev}.png"
    )
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
    losses_by_t_dev: dict[int, list[tuple[float, float]]] = {}
    base, _ = load_scaled_mu_max_comparison(
        base_mu_max,
        SCALING_BASELINE_T_DEV,
        base_mu_max,
        SCALING_BASELINE_T_DEV,
    )
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )

    for comp_mu_max in compare_values:
        comp_t_devs = [t_dev for t_dev in available_t_devs_for_mu(comp_mu_max) if t_dev in TARGET_T_DEVS]
        for t_dev in comp_t_devs:
            _, comp = load_scaled_mu_max_comparison(
                base_mu_max,
                SCALING_BASELINE_T_DEV,
                comp_mu_max,
                t_dev,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            losses_by_t_dev.setdefault(t_dev, []).append((float(comp_mu_max), relative_loss))

    return losses_by_t_dev


def compute_mu_max_loss_table(base_mu_max: float, compare_values: list[float]) -> list[dict[str, float | int]]:
    """
    Build the full table behind the mu_max loss analysis.

    All areas are computed in the fixed scaled space defined by
    SCALING_BASELINE_MU_MAX / SCALING_BASELINE_T_DEV.
    """

    baseline_raw = load_mu_max_border(SCALING_BASELINE_MU_MAX, SCALING_BASELINE_T_DEV)
    scaling_bounds = load_fixed_scaling_bounds()
    scaling_baseline_scaled = normalize_points(baseline_raw, scaling_bounds)
    scaling_baseline_polygon, _ = build_polygon_from_border(
        scaling_baseline_scaled,
        label=(
            f"scaling baseline mu_max={SCALING_BASELINE_MU_MAX:.2f}, "
            f"t_dev={SCALING_BASELINE_T_DEV}"
        ),
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    scaling_baseline_area = geometry_area(scaling_baseline_polygon)

    rows: list[dict[str, float | int]] = []
    base, _ = load_scaled_mu_max_comparison(
        base_mu_max,
        SCALING_BASELINE_T_DEV,
        base_mu_max,
        SCALING_BASELINE_T_DEV,
    )
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    base_area = geometry_area(base_polygon)

    for comp_mu_max in compare_values:
        comp_t_devs = [t_dev for t_dev in available_t_devs_for_mu(comp_mu_max) if t_dev in TARGET_T_DEVS]
        for t_dev in comp_t_devs:
            _, comp = load_scaled_mu_max_comparison(
                base_mu_max,
                SCALING_BASELINE_T_DEV,
                comp_mu_max,
                t_dev,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            lost_area, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            rows.append(
                {
                    "scaling_baseline_mu_max": float(SCALING_BASELINE_MU_MAX),
                    "scaling_baseline_t_dev": int(SCALING_BASELINE_T_DEV),
                    "comparison_baseline_mu_max": float(base_mu_max),
                    "comparison_baseline_t_dev": int(SCALING_BASELINE_T_DEV),
                    "comp_mu_max": float(comp_mu_max),
                    "comp_t_dev": int(t_dev),
                    "q_min": float(scaling_bounds.q_min),
                    "q_max": float(scaling_bounds.q_max),
                    "temp_min": float(scaling_bounds.temp_min),
                    "temp_max": float(scaling_bounds.temp_max),
                    "scaling_baseline_area_scaled": float(scaling_baseline_area),
                    "base_area_scaled": float(base_area),
                    "comp_area_scaled": float(geometry_area(comp_polygon)),
                    "lost_area_scaled": float(lost_area),
                    "relative_loss": float(relative_loss),
                }
            )

    return rows


def save_loss_table_xlsx(rows: list[dict[str, float | int]], out_path: Path) -> Path:
    if not rows:
        raise ValueError("No rows to save.")

    dataframe = pd.DataFrame(rows)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="mu_max_loss", index=False)
    return out_path


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
    ax.set_title(
        f"Scaled area loss by mu_max\nbaseline mu_max={SCALING_BASELINE_MU_MAX:.2f}, "
        f"year {t_dev_to_year(SCALING_BASELINE_T_DEV)}"
    )
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "mu_max_relative_loss.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_delay_compensation_example(
    baseline: BorderScenario,
    delayed: BorderScenario,
    compensated: BorderScenario,
) -> tuple[Path, dict[str, float]]:
    baseline_polygon, _ = build_polygon_from_border(
        baseline.scaled_points,
        label=f"baseline mu_max={baseline.sweep_value:.2f}, t_dev={baseline.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    delayed_polygon, _ = build_polygon_from_border(
        delayed.scaled_points,
        label=f"delayed mu_max={delayed.sweep_value:.2f}, t_dev={delayed.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    compensated_polygon, _ = build_polygon_from_border(
        compensated.scaled_points,
        label=f"compensated mu_max={compensated.sweep_value:.2f}, t_dev={compensated.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )

    delayed_lost_area, delayed_relative_loss, delayed_lost_geometry = relative_area_loss(
        baseline_polygon, delayed_polygon
    )
    compensated_lost_area, compensated_relative_loss, _ = relative_area_loss(
        baseline_polygon, compensated_polygon
    )
    recovered_loss_geometry = delayed_lost_geometry.intersection(compensated_polygon)
    recovered_loss_area = geometry_area(recovered_loss_geometry)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    baseline_color = "tab:blue"
    delayed_color = "tab:orange"
    compensated_color = "tab:green"
    recovered_fill = "#1b9e77"

    _plot_border(
        ax,
        baseline.scaled_points,
        f"Baseline mu_max={baseline.sweep_value:.2f}, year {t_dev_to_year(baseline.t_dev)}",
        baseline_color,
    )
    _plot_border(
        ax,
        delayed.scaled_points,
        f"Delayed mitigation to {t_dev_to_year(delayed.t_dev)}",
        delayed_color,
    )
    _plot_border(
        ax,
        compensated.scaled_points,
        f"Compensated to mu_max={compensated.sweep_value:.2f}, year {t_dev_to_year(compensated.t_dev)}",
        compensated_color,
    )

    recovered_labeled = False
    for coords in iter_polygon_patches(recovered_loss_geometry):
        ax.fill(
            coords[:, 0],
            coords[:, 1],
            color=recovered_fill,
            alpha=0.24,
            label="Recovered baseline loss" if not recovered_labeled else None,
        )
        recovered_labeled = True

    ax.set_xlabel("Scaled Q")
    ax.set_ylabel("Scaled Delta T")
    ax.set_title(
        "Delayed mitigation vs compensation\n"
        f"baseline mu_max={baseline.sweep_value:.2f}, year {t_dev_to_year(baseline.t_dev)}"
    )
    ax.set_xlim(*common_scaled_axis_limits()[:2])
    ax.set_ylim(*common_scaled_axis_limits()[2:])
    ax.grid(True)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        (
            f"Delay loss = {delayed_relative_loss:.4f}\n"
            f"After compensation = {compensated_relative_loss:.4f}\n"
            f"Recovered loss = {recovered_loss_area:.4f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    plt.tight_layout()

    out_path = OUTPUT_DIR / (
        f"delay_compensation_mu{baseline.sweep_value:.2f}_tdev{baseline.t_dev}"
        f"_delay{delayed.t_dev}_compmu{compensated.sweep_value:.2f}.png"
    )
    plt.savefig(out_path)
    plt.close(fig)

    return out_path, {
        "baseline_area_scaled": geometry_area(baseline_polygon),
        "delayed_area_scaled": geometry_area(delayed_polygon),
        "compensated_area_scaled": geometry_area(compensated_polygon),
        "delayed_lost_area_scaled": delayed_lost_area,
        "delayed_relative_loss": delayed_relative_loss,
        "compensated_lost_area_scaled": compensated_lost_area,
        "compensated_relative_loss": compensated_relative_loss,
        "recovered_loss_area_scaled": recovered_loss_area,
    }


def run_mu_12_tdev_series() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("mu=1.2 t_dev series requires shapely.")

    available_t_devs = available_t_devs_for_mu(SCALING_BASELINE_MU_MAX)

    for comp_t_dev in available_t_devs:
        base, comp = load_scaled_mu_max_comparison(
            SCALING_BASELINE_MU_MAX,
            SCALING_BASELINE_T_DEV,
            SCALING_BASELINE_MU_MAX,
            comp_t_dev,
        )
        lost_region_path, debug_stats = plot_lost_region(base, comp, annotate_loss=0.0)
        base_polygon, _ = build_polygon_from_border(
            base.scaled_points,
            label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
            repair_invalid=REPAIR_INVALID_POLYGONS,
        )
        comp_polygon, _ = build_polygon_from_border(
            comp.scaled_points,
            label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
            repair_invalid=REPAIR_INVALID_POLYGONS,
        )
        _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
        final_path, _ = plot_lost_region(base, comp, annotate_loss=relative_loss)
        print(
            "Saved mu=1.2 t_dev comparison: "
            f"baseline t_dev={SCALING_BASELINE_T_DEV} vs comp t_dev={comp_t_dev} -> {final_path}"
        )
        if final_path != lost_region_path:
            print(f"Updated preliminary loss-region plot: {lost_region_path}")
        print(
            f"  base_area={debug_stats['base_area_scaled']:.6f}, "
            f"comp_area={debug_stats['comp_area_scaled']:.6f}, "
            f"lost_area={debug_stats['lost_area_scaled']:.6f}, "
            f"relative_loss={relative_loss:.6f}"
        )


def run_mu_max_loss_regions() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("mu_max loss-region generation requires shapely.")

    for comp_mu_max in COMPARE_MU_MAX_VALUES:
        for comp_t_dev in TARGET_T_DEVS:
            try:
                base, comp = load_scaled_mu_max_comparison(
                    SCALING_BASELINE_MU_MAX,
                    SCALING_BASELINE_T_DEV,
                    comp_mu_max,
                    comp_t_dev,
                )
            except FileNotFoundError:
                continue

            base_polygon, _ = build_polygon_from_border(
                base.scaled_points,
                label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            out_path, debug_stats = plot_lost_region(base, comp, annotate_loss=relative_loss)
            print(
                "Saved mu_max loss-region plot: "
                f"baseline mu={base.sweep_value:.2f}, t_dev={base.t_dev} vs "
                f"comp mu={comp.sweep_value:.2f}, t_dev={comp.t_dev} -> {out_path}"
            )
            print(
                f"  base_area={debug_stats['base_area_scaled']:.6f}, "
                f"comp_area={debug_stats['comp_area_scaled']:.6f}, "
                f"lost_area={debug_stats['lost_area_scaled']:.6f}, "
                f"relative_loss={relative_loss:.6f}"
            )


def run_delay_compensation_series() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("delay compensation series requires shapely.")

    for comp_mu_max in COMPARE_MU_MAX_VALUES:
        if comp_mu_max <= SCALING_BASELINE_MU_MAX + 1e-12:
            continue
        for comp_t_dev in TARGET_T_DEVS:
            try:
                baseline, delayed = load_scaled_mu_max_comparison(
                    SCALING_BASELINE_MU_MAX,
                    SCALING_BASELINE_T_DEV,
                    SCALING_BASELINE_MU_MAX,
                    comp_t_dev,
                )
                _, compensated = load_scaled_mu_max_comparison(
                    SCALING_BASELINE_MU_MAX,
                    SCALING_BASELINE_T_DEV,
                    comp_mu_max,
                    comp_t_dev,
                )
            except FileNotFoundError:
                continue

            out_path, stats = plot_delay_compensation_example(baseline, delayed, compensated)
            print(
                "Saved delay-compensation series plot: "
                f"baseline mu={baseline.sweep_value:.2f}, t_dev={baseline.t_dev}; "
                f"delayed t_dev={delayed.t_dev}; "
                f"comp mu={compensated.sweep_value:.2f}, t_dev={compensated.t_dev} -> {out_path}"
            )
            print(
                f"  delay_loss={stats['delayed_relative_loss']:.6f}, "
                f"after_compensation={stats['compensated_relative_loss']:.6f}, "
                f"recovered_loss={stats['recovered_loss_area_scaled']:.6f}"
            )


def run_delay_compensation_example() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("delay compensation example requires shapely.")

    baseline, delayed = load_scaled_mu_max_comparison(
        SCALING_BASELINE_MU_MAX,
        SCALING_BASELINE_T_DEV,
        SCALING_BASELINE_MU_MAX,
        DEMO_T_DEV,
    )
    _, compensated = load_scaled_mu_max_comparison(
        SCALING_BASELINE_MU_MAX,
        SCALING_BASELINE_T_DEV,
        DEMO_COMPARED_MU_MAX,
        DEMO_T_DEV,
    )
    out_path, stats = plot_delay_compensation_example(baseline, delayed, compensated)
    print(f"Saved delay-compensation example plot: {out_path}")
    print(
        f"  delay_loss={stats['delayed_relative_loss']:.6f}, "
        f"after_compensation={stats['compensated_relative_loss']:.6f}, "
        f"recovered_loss={stats['recovered_loss_area_scaled']:.6f}"
    )


def run_demo() -> None:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("Demo requires shapely.")

    base, comp = load_scaled_mu_max_comparison(
        SCALING_BASELINE_MU_MAX,
        SCALING_BASELINE_T_DEV,
        DEMO_COMPARED_MU_MAX,
        DEMO_T_DEV,
    )
    scaling_bounds = load_fixed_scaling_bounds()
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
        "Fixed scaling baseline: "
        f"mu_max={SCALING_BASELINE_MU_MAX:.2f}, "
        f"t_dev={SCALING_BASELINE_T_DEV}, "
        f"q=[{scaling_bounds.q_min:.6f}, {scaling_bounds.q_max:.6f}], "
        f"T=[{scaling_bounds.temp_min:.6f}, {scaling_bounds.temp_max:.6f}]"
    )
    print(
        "Demo comparison: "
        f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev} vs "
        f"compensated mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}"
    )
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
    rows = compute_mu_max_loss_table(BASELINE_MU_MAX, COMPARE_MU_MAX_VALUES)
    losses_by_t_dev = compute_mu_max_loss_series(BASELINE_MU_MAX, COMPARE_MU_MAX_VALUES)
    xlsx_path = save_loss_table_xlsx(rows, OUTPUT_DIR / "mu_max_loss_table.xlsx")
    out_path = plot_mu_max_loss_series(losses_by_t_dev)
    print(f"Saved mu_max loss table: {xlsx_path}")
    print(f"Saved mu_max loss plot: {out_path}")
    for row in rows:
        print(
            "row: "
            f"base_mu={row['comparison_baseline_mu_max']:.2f}, "
            f"base_t_dev={row['comparison_baseline_t_dev']}, "
            f"comp_mu={row['comp_mu_max']:.2f}, "
            f"comp_t_dev={row['comp_t_dev']}, "
            f"base_area={row['base_area_scaled']:.6f}, "
            f"comp_area={row['comp_area_scaled']:.6f}, "
            f"lost_area={row['lost_area_scaled']:.6f}, "
            f"relative_loss={row['relative_loss']:.6f}"
        )
    for t_dev, values in sorted(losses_by_t_dev.items()):
        print(f"t_dev={t_dev}: {values}")


if __name__ == "__main__":
    if RUN_DEMO:
        run_demo()
    if RUN_MU_MAX_ANALYSIS:
        run_mu_max_analysis()
    if RUN_MU_12_TDEV_SERIES:
        run_mu_12_tdev_series()
    if RUN_MU_MAX_LOSS_REGIONS:
        run_mu_max_loss_regions()
    if RUN_DELAY_COMPENSATION_EXAMPLE:
        run_delay_compensation_example()
    if RUN_DELAY_COMPENSATION_SERIES:
        run_delay_compensation_series()
