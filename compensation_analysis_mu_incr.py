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


OUTPUT_DIR = Path("plots_png") / "compensation_muincr"
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

BASELINE_MU_INCR = 1.1
SCALING_BASELINE_MU_INCR = 1.1
SCALING_BASELINE_T_DEV = 1
COMPARE_MU_INCR_VALUES = [1.1, 1.2, 1.3, 1.4, 1.5]
DELAY_COMPENSATION_MU_INCR_VALUES = [1.2, 1.3, 1.4, 1.5, sc.MU_INCR_NO_CONSTRAINT]
TARGET_T_DEVS = [5, 6, 7]
DEMO_COMPARED_MU_INCR = 1.2
DEMO_T_DEV = 5


@dataclass(frozen=True)
class BorderScenario:
    scenario_name: str
    sweep_value: float | str
    t_dev: int
    raw_points: np.ndarray
    scaled_points: np.ndarray


def t_dev_to_year(t_dev: int) -> int:
    return 2015 + 5 * int(t_dev)


def _load_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got {points.shape[1]}.")
    return points


def format_muincr(mu_incr: float | str) -> str:
    if isinstance(mu_incr, str):
        return mu_incr
    return f"{float(mu_incr):.2f}"


def load_muincr_border(mu_incr: float | str, t_dev: int) -> np.ndarray:
    path = Path(sc.data_filename(sc.SCENARIO_CHANGE_MU_INCR, mu_incr, t_dev))
    if not path.exists():
        raise FileNotFoundError(f"Missing border file: {path}")
    return _load_points(path)


def load_fixed_scaling_bounds() -> ScalingBounds:
    baseline_raw = load_muincr_border(SCALING_BASELINE_MU_INCR, SCALING_BASELINE_T_DEV)
    return ScalingBounds(
        q_min=float(np.min(baseline_raw[:, 0])),
        q_max=float(np.max(baseline_raw[:, 0])),
        temp_min=float(np.min(baseline_raw[:, 1])),
        temp_max=float(np.max(baseline_raw[:, 1])),
    )


def available_t_devs_for_muincr(mu_incr: float) -> list[int]:
    available = []
    for t_dev in TARGET_T_DEVS + [SCALING_BASELINE_T_DEV]:
        try:
            load_muincr_border(mu_incr, t_dev)
            available.append(t_dev)
        except FileNotFoundError:
            continue
    return sorted(set(available))


@lru_cache(maxsize=1)
def common_scaled_axis_limits() -> tuple[float, float, float, float]:
    bounds = load_fixed_scaling_bounds()
    all_points = []
    axis_values = list(COMPARE_MU_INCR_VALUES) + [sc.MU_INCR_NO_CONSTRAINT]
    for mu_incr in axis_values:
        for t_dev in available_t_devs_for_muincr(mu_incr):
            try:
                raw_points = load_muincr_border(mu_incr, t_dev)
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


def load_scaled_muincr_comparison(
    base_mu_incr: float | str,
    base_t_dev: int,
    comp_mu_incr: float | str,
    comp_t_dev: int,
) -> tuple[BorderScenario, BorderScenario]:
    bounds = load_fixed_scaling_bounds()
    base_raw = load_muincr_border(base_mu_incr, base_t_dev)
    comp_raw = load_muincr_border(comp_mu_incr, comp_t_dev)

    base = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_INCR,
        sweep_value=base_mu_incr,
        t_dev=base_t_dev,
        raw_points=base_raw,
        scaled_points=normalize_points(base_raw, bounds),
    )
    comp = BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_INCR,
        sweep_value=comp_mu_incr,
        t_dev=comp_t_dev,
        raw_points=comp_raw,
        scaled_points=normalize_points(comp_raw, bounds),
    )
    return base, comp


def _scenario_role_label(base: BorderScenario, comp: BorderScenario) -> str:
    if (
        not isinstance(comp.sweep_value, str)
        and not isinstance(base.sweep_value, str)
        and np.isclose(float(comp.sweep_value), float(base.sweep_value))
        and comp.t_dev != base.t_dev
    ):
        return f"Delayed mitigation to {t_dev_to_year(comp.t_dev)}"
    if comp.sweep_value == sc.MU_INCR_NO_CONSTRAINT:
        return f"Compensated to mu_incr=none, year {t_dev_to_year(comp.t_dev)}"
    if not isinstance(comp.sweep_value, str) and not isinstance(base.sweep_value, str) and float(comp.sweep_value) > float(base.sweep_value):
        return f"Compensated to mu_incr={float(comp.sweep_value):.2f}, year {t_dev_to_year(comp.t_dev)}"
    return f"Comparison year {t_dev_to_year(comp.t_dev)}"


def _plot_border(ax, points: np.ndarray, label: str, color: str) -> None:
    closed = np.vstack([points, points[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.6, label=label)


def plot_lost_region(base: BorderScenario, comp: BorderScenario, *, annotate_loss: float) -> tuple[Path, dict[str, float]]:
    base_polygon, base_debug = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_incr={format_muincr(base.sweep_value)}, t_dev={base.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    comp_polygon, comp_debug = build_polygon_from_border(
        comp.scaled_points,
        label=f"comp mu_incr={format_muincr(comp.sweep_value)}, t_dev={comp.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    lost_area, relative_loss, lost_geometry = relative_area_loss(base_polygon, comp_polygon)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    _plot_border(
        ax,
        base.scaled_points,
        f"Baseline mu_incr={format_muincr(base.sweep_value)}, year {t_dev_to_year(base.t_dev)}",
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
        f"loss_region_muincr{format_muincr(base.sweep_value)}_tdev{base.t_dev}"
        f"_vs_muincr{format_muincr(comp.sweep_value)}_tdev{comp.t_dev}.png"
    )
    plt.savefig(out_path)
    plt.close(fig)

    return out_path, {
        "base_area_scaled": geometry_area(base_polygon),
        "comp_area_scaled": geometry_area(comp_polygon),
        "lost_area_scaled": lost_area,
        "relative_loss": relative_loss,
        "base_signed_area": base_debug.signed_area,
        "comp_signed_area": comp_debug.signed_area,
    }


def plot_delay_compensation_example(
    baseline: BorderScenario,
    delayed: BorderScenario,
    compensated: BorderScenario,
) -> tuple[Path, dict[str, float]]:
    baseline_polygon, _ = build_polygon_from_border(
        baseline.scaled_points,
        label=f"baseline mu_incr={format_muincr(baseline.sweep_value)}, t_dev={baseline.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    delayed_polygon, _ = build_polygon_from_border(
        delayed.scaled_points,
        label=f"delayed mu_incr={format_muincr(delayed.sweep_value)}, t_dev={delayed.t_dev}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    compensated_polygon, _ = build_polygon_from_border(
        compensated.scaled_points,
        label=f"compensated mu_incr={format_muincr(compensated.sweep_value)}, t_dev={compensated.t_dev}",
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
    _plot_border(
        ax,
        baseline.scaled_points,
        f"Baseline mu_incr={format_muincr(baseline.sweep_value)}, year {t_dev_to_year(baseline.t_dev)}",
        "tab:blue",
    )
    _plot_border(
        ax,
        delayed.scaled_points,
        f"Delayed mitigation to {t_dev_to_year(delayed.t_dev)}",
        "tab:orange",
    )
    _plot_border(
        ax,
        compensated.scaled_points,
        f"Compensated to mu_incr={format_muincr(compensated.sweep_value)}, year {t_dev_to_year(compensated.t_dev)}",
        "tab:green",
    )

    recovered_labeled = False
    for coords in iter_polygon_patches(recovered_loss_geometry):
        ax.fill(
            coords[:, 0],
            coords[:, 1],
            color="#1b9e77",
            alpha=0.24,
            label="Recovered baseline loss" if not recovered_labeled else None,
        )
        recovered_labeled = True

    ax.set_xlabel("Scaled Q")
    ax.set_ylabel("Scaled Delta T")
    ax.set_title(
        "Delayed mitigation vs compensation\n"
        f"baseline mu_incr={format_muincr(baseline.sweep_value)}, year {t_dev_to_year(baseline.t_dev)}"
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
        f"delay_compensation_muincr{format_muincr(baseline.sweep_value)}_tdev{baseline.t_dev}"
        f"_delay{delayed.t_dev}_compincr{format_muincr(compensated.sweep_value)}.png"
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


def compute_muincr_loss_series(base_mu_incr: float, compare_values: list[float]) -> dict[int, list[tuple[float, float]]]:
    losses_by_t_dev: dict[int, list[tuple[float, float]]] = {}
    base, _ = load_scaled_muincr_comparison(
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
    )
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_incr={format_muincr(base.sweep_value)}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )

    for comp_mu_incr in compare_values:
        comp_t_devs = [t_dev for t_dev in available_t_devs_for_muincr(comp_mu_incr) if t_dev in TARGET_T_DEVS]
        for t_dev in comp_t_devs:
            _, comp = load_scaled_muincr_comparison(
                base_mu_incr,
                SCALING_BASELINE_T_DEV,
                comp_mu_incr,
                t_dev,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_incr={format_muincr(comp.sweep_value)}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            losses_by_t_dev.setdefault(t_dev, []).append((float(comp_mu_incr), relative_loss))

    return losses_by_t_dev


def compute_muincr_loss_series_with_none(
    base_mu_incr: float,
    compare_values: list[float],
) -> dict[int, list[tuple[float | str, float]]]:
    losses_by_t_dev = compute_muincr_loss_series(base_mu_incr, compare_values)

    base, _ = load_scaled_muincr_comparison(
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
    )
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_incr={format_muincr(base.sweep_value)}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )

    none_t_devs = [t_dev for t_dev in available_t_devs_for_muincr(sc.MU_INCR_NO_CONSTRAINT) if t_dev in TARGET_T_DEVS]
    for t_dev in none_t_devs:
        _, comp = load_scaled_muincr_comparison(
            base_mu_incr,
            SCALING_BASELINE_T_DEV,
            sc.MU_INCR_NO_CONSTRAINT,
            t_dev,
        )
        comp_polygon, _ = build_polygon_from_border(
            comp.scaled_points,
            label=f"comp mu_incr={format_muincr(comp.sweep_value)}, t_dev={t_dev}",
            repair_invalid=REPAIR_INVALID_POLYGONS,
        )
        _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
        losses_by_t_dev.setdefault(t_dev, []).append((sc.MU_INCR_NO_CONSTRAINT, float(relative_loss)))

    return losses_by_t_dev


def compute_muincr_loss_table(base_mu_incr: float, compare_values: list[float]) -> list[dict[str, float | int]]:
    baseline_raw = load_muincr_border(SCALING_BASELINE_MU_INCR, SCALING_BASELINE_T_DEV)
    scaling_bounds = load_fixed_scaling_bounds()
    scaling_baseline_scaled = normalize_points(baseline_raw, scaling_bounds)
    scaling_baseline_polygon, _ = build_polygon_from_border(
        scaling_baseline_scaled,
        label=f"scaling baseline mu_incr={SCALING_BASELINE_MU_INCR:.2f}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    scaling_baseline_area = geometry_area(scaling_baseline_polygon)

    rows: list[dict[str, float | int]] = []
    base, _ = load_scaled_muincr_comparison(
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
        base_mu_incr,
        SCALING_BASELINE_T_DEV,
    )
    base_polygon, _ = build_polygon_from_border(
        base.scaled_points,
        label=f"baseline mu_incr={format_muincr(base.sweep_value)}, t_dev={SCALING_BASELINE_T_DEV}",
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    base_area = geometry_area(base_polygon)

    for comp_mu_incr in compare_values:
        comp_t_devs = [t_dev for t_dev in available_t_devs_for_muincr(comp_mu_incr) if t_dev in TARGET_T_DEVS]
        for t_dev in comp_t_devs:
            _, comp = load_scaled_muincr_comparison(
                base_mu_incr,
                SCALING_BASELINE_T_DEV,
                comp_mu_incr,
                t_dev,
            )
            comp_polygon, _ = build_polygon_from_border(
                comp.scaled_points,
                label=f"comp mu_incr={format_muincr(comp.sweep_value)}, t_dev={t_dev}",
                repair_invalid=REPAIR_INVALID_POLYGONS,
            )
            lost_area, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
            rows.append(
                {
                    "scaling_baseline_mu_incr": float(SCALING_BASELINE_MU_INCR),
                    "scaling_baseline_t_dev": int(SCALING_BASELINE_T_DEV),
                    "comparison_baseline_mu_incr": float(base_mu_incr),
                    "comparison_baseline_t_dev": int(SCALING_BASELINE_T_DEV),
                    "comp_mu_incr": float(comp_mu_incr),
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
        dataframe.to_excel(writer, sheet_name="mu_incr_loss", index=False)
    return out_path


def plot_muincr_loss_series(losses_by_t_dev: dict[int, list[tuple[float, float]]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    x_order: list[float | str] = [1.1, 1.2, 1.3, 1.4, 1.5, sc.MU_INCR_NO_CONSTRAINT]
    x_positions = {value: idx for idx, value in enumerate(x_order)}
    for t_dev, values in sorted(losses_by_t_dev.items()):
        values = sorted(values, key=lambda item: x_positions[item[0]])
        ax.plot(
            [x_positions[item[0]] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=1.8,
            label=f"Delay to {t_dev_to_year(t_dev)}",
        )

    ax.set_xlabel("mu_incr")
    ax.set_ylabel("Relative loss")
    ax.set_title(
        f"Scaled area loss by mu_incr\nbaseline mu_incr={SCALING_BASELINE_MU_INCR:.2f}, "
        f"year {t_dev_to_year(SCALING_BASELINE_T_DEV)}"
    )
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(["1.1", "1.2", "1.3", "1.4", "1.5", "none"])
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "mu_incr_relative_loss.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def run_muincr_analysis() -> None:
    rows = compute_muincr_loss_table(BASELINE_MU_INCR, COMPARE_MU_INCR_VALUES)
    losses_by_t_dev = compute_muincr_loss_series_with_none(BASELINE_MU_INCR, COMPARE_MU_INCR_VALUES)
    xlsx_path = save_loss_table_xlsx(rows, OUTPUT_DIR / "mu_incr_loss_table.xlsx")
    plot_path = plot_muincr_loss_series(losses_by_t_dev)
    print(f"Saved mu_incr loss table: {xlsx_path}")
    print(f"Saved mu_incr loss plot: {plot_path}")


def run_muincr_loss_regions() -> None:
    for comp_t_dev in [t_dev for t_dev in available_t_devs_for_muincr(SCALING_BASELINE_MU_INCR) if t_dev in TARGET_T_DEVS]:
        base, comp = load_scaled_muincr_comparison(
            SCALING_BASELINE_MU_INCR,
            SCALING_BASELINE_T_DEV,
            SCALING_BASELINE_MU_INCR,
            comp_t_dev,
        )
        base_polygon, _ = build_polygon_from_border(base.scaled_points, label="base", repair_invalid=REPAIR_INVALID_POLYGONS)
        comp_polygon, _ = build_polygon_from_border(comp.scaled_points, label="comp", repair_invalid=REPAIR_INVALID_POLYGONS)
        _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
        out_path, stats = plot_lost_region(base, comp, annotate_loss=relative_loss)
        print(f"Saved mu_incr loss-region plot: {out_path}")
        print(
            f"  base_area={stats['base_area_scaled']:.6f}, "
            f"comp_area={stats['comp_area_scaled']:.6f}, "
            f"lost_area={stats['lost_area_scaled']:.6f}, "
            f"relative_loss={relative_loss:.6f}"
        )


def run_delay_compensation_series() -> None:
    for comp_mu_incr in DELAY_COMPENSATION_MU_INCR_VALUES:
        for comp_t_dev in TARGET_T_DEVS:
            try:
                baseline, delayed = load_scaled_muincr_comparison(
                    SCALING_BASELINE_MU_INCR,
                    SCALING_BASELINE_T_DEV,
                    SCALING_BASELINE_MU_INCR,
                    comp_t_dev,
                )
                _, compensated = load_scaled_muincr_comparison(
                    SCALING_BASELINE_MU_INCR,
                    SCALING_BASELINE_T_DEV,
                    comp_mu_incr,
                    comp_t_dev,
                )
            except FileNotFoundError:
                continue

            out_path, stats = plot_delay_compensation_example(baseline, delayed, compensated)
            print(f"Saved mu_incr delay-compensation plot: {out_path}")
            print(
                f"  delay_loss={stats['delayed_relative_loss']:.6f}, "
                f"after_compensation={stats['compensated_relative_loss']:.6f}, "
                f"recovered_loss={stats['recovered_loss_area_scaled']:.6f}"
            )


if __name__ == "__main__":
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("This script requires shapely.")
    run_muincr_analysis()
    run_muincr_loss_regions()
    run_delay_compensation_series()
