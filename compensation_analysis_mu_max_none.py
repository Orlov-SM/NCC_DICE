from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compensation_analysis as ca
import scenario_config as sc
from geometry_utils import build_polygon_from_border, geometry_area, iter_polygon_patches, relative_area_loss
from scaling_utils import ScalingBounds, normalize_points


ca.OUTPUT_DIR = Path("plots_png") / "compensation_mu_max_no_constraint"
ca.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ca.COMPARE_MU_MAX_VALUES = [round(value, 4) for value in np.linspace(1.2, 1.5, 10)]


def compensation_mu_values() -> list[float]:
    return [
        float(mu_max)
        for mu_max in ca.COMPARE_MU_MAX_VALUES
        if float(mu_max) > ca.SCALING_BASELINE_MU_MAX + 1e-12
    ]


def _load_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got {points.shape[1]}.")
    return points


def load_standard_mu_max_border(mu_max: float, t_dev: int) -> np.ndarray:
    path = Path(sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX, mu_max, t_dev))
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline border file: {path}")
    return _load_points(path)


def load_mu_max_no_constraint_border(mu_max: float, t_dev: int) -> np.ndarray:
    path = Path(sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT, mu_max, t_dev))
    if not path.exists():
        raise FileNotFoundError(f"Missing no-constraint border file: {path}")
    return _load_points(path)


def load_fixed_scaling_bounds() -> ScalingBounds:
    baseline_raw = load_standard_mu_max_border(ca.SCALING_BASELINE_MU_MAX, ca.SCALING_BASELINE_T_DEV)
    return ScalingBounds(
        q_min=float(np.min(baseline_raw[:, 0])),
        q_max=float(np.max(baseline_raw[:, 0])),
        temp_min=float(np.min(baseline_raw[:, 1])),
        temp_max=float(np.max(baseline_raw[:, 1])),
    )


def available_t_devs_for_mu(mu_max: float) -> list[int]:
    available = []
    for t_dev in ca.TARGET_T_DEVS + [ca.SCALING_BASELINE_T_DEV]:
        try:
            load_mu_max_no_constraint_border(mu_max, t_dev)
            available.append(t_dev)
        except FileNotFoundError:
            continue
    return sorted(set(available))


@lru_cache(maxsize=1)
def common_scaled_axis_limits() -> tuple[float, float, float, float]:
    bounds = load_fixed_scaling_bounds()
    all_points = [normalize_points(load_standard_mu_max_border(ca.SCALING_BASELINE_MU_MAX, ca.SCALING_BASELINE_T_DEV), bounds)]
    for mu_max in ca.COMPARE_MU_MAX_VALUES:
        for t_dev in available_t_devs_for_mu(mu_max):
            try:
                raw_points = load_mu_max_no_constraint_border(mu_max, t_dev)
            except FileNotFoundError:
                continue
            all_points.append(normalize_points(raw_points, bounds))

    merged = np.vstack(all_points)
    x_min, x_max = float(np.min(merged[:, 0])), float(np.max(merged[:, 0]))
    y_min, y_max = float(np.min(merged[:, 1])), float(np.max(merged[:, 1]))
    x_pad = max(0.03, 0.05 * (x_max - x_min))
    y_pad = max(0.03, 0.05 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


@lru_cache(maxsize=1)
def common_raw_axis_limits() -> tuple[float, float, float, float]:
    all_points = [load_standard_mu_max_border(ca.SCALING_BASELINE_MU_MAX, ca.SCALING_BASELINE_T_DEV)]
    for mu_max in ca.COMPARE_MU_MAX_VALUES:
        for t_dev in available_t_devs_for_mu(mu_max):
            try:
                all_points.append(load_mu_max_no_constraint_border(mu_max, t_dev))
            except FileNotFoundError:
                continue

    merged = np.vstack(all_points)
    x_min, x_max = float(np.min(merged[:, 0])), float(np.max(merged[:, 0]))
    y_min, y_max = float(np.min(merged[:, 1])), float(np.max(merged[:, 1]))
    x_pad = max(10.0, 0.05 * (x_max - x_min))
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad, max(0.0, y_min - y_pad), y_max + y_pad)


def load_scaled_mu_max_comparison(
    base_mu_max: float,
    base_t_dev: int,
    comp_mu_max: float,
    comp_t_dev: int,
) -> tuple[ca.BorderScenario, ca.BorderScenario]:
    bounds = load_fixed_scaling_bounds()
    base_raw = load_standard_mu_max_border(base_mu_max, base_t_dev)
    comp_raw = load_mu_max_no_constraint_border(comp_mu_max, comp_t_dev)

    base = ca.BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=base_mu_max,
        t_dev=base_t_dev,
        raw_points=base_raw,
        scaled_points=normalize_points(base_raw, bounds),
    )
    comp = ca.BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT,
        sweep_value=comp_mu_max,
        t_dev=comp_t_dev,
        raw_points=comp_raw,
        scaled_points=normalize_points(comp_raw, bounds),
    )
    return base, comp


def load_scaled_mu_max_pair(base_mu_max: float, comp_mu_max: float, t_dev: int) -> tuple[ca.BorderScenario, ca.BorderScenario]:
    return load_scaled_mu_max_comparison(base_mu_max, t_dev, comp_mu_max, t_dev)


def load_scaled_standard_delay(
    base_mu_max: float,
    base_t_dev: int,
    delayed_t_dev: int,
) -> tuple[ca.BorderScenario, ca.BorderScenario]:
    bounds = load_fixed_scaling_bounds()
    base_raw = load_standard_mu_max_border(base_mu_max, base_t_dev)
    delayed_raw = load_standard_mu_max_border(base_mu_max, delayed_t_dev)

    base = ca.BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=base_mu_max,
        t_dev=base_t_dev,
        raw_points=base_raw,
        scaled_points=normalize_points(base_raw, bounds),
    )
    delayed = ca.BorderScenario(
        scenario_name=sc.SCENARIO_CHANGE_MU_MAX,
        sweep_value=base_mu_max,
        t_dev=delayed_t_dev,
        raw_points=delayed_raw,
        scaled_points=normalize_points(delayed_raw, bounds),
    )
    return base, delayed


def _scenario_role_label(base: ca.BorderScenario, comp: ca.BorderScenario) -> str:
    if np.isclose(float(comp.sweep_value), float(base.sweep_value)) and comp.t_dev != base.t_dev:
        return f"Delayed mitigation to {ca.t_dev_to_year(comp.t_dev)}"
    if float(comp.sweep_value) > float(base.sweep_value):
        return (
            f"Compensated to mu_max={float(comp.sweep_value):.2f}, "
            f"year {ca.t_dev_to_year(comp.t_dev)} (no mu_incr constraint)"
        )
    return f"Comparison year {ca.t_dev_to_year(comp.t_dev)} (no mu_incr constraint)"


def draw_delay_compensation_panel(
    ax,
    baseline: ca.BorderScenario,
    delayed: ca.BorderScenario,
    compensated: ca.BorderScenario,
    *,
    show_title: bool = True,
    annotation_xy: tuple[float, float] = (0.02, 0.98),
    annotation_ha: str = "left",
    annotation_va: str = "top",
) -> dict[str, float]:
    baseline_polygon, _ = build_polygon_from_border(
        baseline.scaled_points,
        label=f"baseline mu_max={baseline.sweep_value:.2f}, t_dev={baseline.t_dev}",
        repair_invalid=ca.REPAIR_INVALID_POLYGONS,
    )
    delayed_polygon, _ = build_polygon_from_border(
        delayed.scaled_points,
        label=f"delayed mu_max={delayed.sweep_value:.2f}, t_dev={delayed.t_dev}",
        repair_invalid=ca.REPAIR_INVALID_POLYGONS,
    )
    compensated_polygon, _ = build_polygon_from_border(
        compensated.scaled_points,
        label=f"compensated mu_max={compensated.sweep_value:.2f}, t_dev={compensated.t_dev}, no constraint",
        repair_invalid=ca.REPAIR_INVALID_POLYGONS,
    )

    delayed_lost_area, delayed_relative_loss, delayed_lost_geometry = relative_area_loss(
        baseline_polygon, delayed_polygon
    )
    compensated_lost_area, compensated_relative_loss, _ = relative_area_loss(
        baseline_polygon, compensated_polygon
    )
    recovered_loss_geometry = delayed_lost_geometry.intersection(compensated_polygon)
    recovered_loss_area = geometry_area(recovered_loss_geometry)

    ca._plot_border(
        ax,
        baseline.scaled_points,
        f"Baseline mu_max={baseline.sweep_value:.2f}, year {ca.t_dev_to_year(baseline.t_dev)}",
        "tab:blue",
    )
    ca._plot_border(
        ax,
        delayed.scaled_points,
        f"Delayed mitigation to {ca.t_dev_to_year(delayed.t_dev)}",
        "tab:orange",
    )
    ca._plot_border(
        ax,
        compensated.scaled_points,
        (
            f"Compensated to mu_max={compensated.sweep_value:.2f}, "
            f"year {ca.t_dev_to_year(compensated.t_dev)} (no mu_incr constraint)"
        ),
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
    if show_title:
        ax.set_title(
            "Delayed mitigation vs compensation\n"
            f"baseline mu_max={baseline.sweep_value:.2f}, year {ca.t_dev_to_year(baseline.t_dev)}"
        )
    ax.set_xlim(*common_scaled_axis_limits()[:2])
    ax.set_ylim(*common_scaled_axis_limits()[2:])
    ax.grid(True)
    ax.legend(loc="best")
    ax.text(
        annotation_xy[0],
        annotation_xy[1],
        (
            f"Delay loss = {delayed_relative_loss:.4f}\n"
            f"After compensation = {compensated_relative_loss:.4f}\n"
            f"Recovered loss = {recovered_loss_area:.4f}"
        ),
        transform=ax.transAxes,
        va=annotation_va,
        ha=annotation_ha,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    return {
        "baseline_area_scaled": geometry_area(baseline_polygon),
        "delayed_area_scaled": geometry_area(delayed_polygon),
        "compensated_area_scaled": geometry_area(compensated_polygon),
        "delayed_lost_area_scaled": delayed_lost_area,
        "delayed_relative_loss": delayed_relative_loss,
        "compensated_lost_area_scaled": compensated_lost_area,
        "compensated_relative_loss": compensated_relative_loss,
        "recovered_loss_area_scaled": recovered_loss_area,
    }


def plot_mu_max_loss_series(losses_by_t_dev: dict[int, list[tuple[float, float]]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for t_dev, values in sorted(losses_by_t_dev.items()):
        baseline, delayed = load_scaled_standard_delay(
            ca.SCALING_BASELINE_MU_MAX,
            ca.SCALING_BASELINE_T_DEV,
            t_dev,
        )
        baseline_polygon, _ = build_polygon_from_border(
            baseline.scaled_points,
            label=f"baseline mu_max={baseline.sweep_value:.2f}, t_dev={baseline.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        delayed_polygon, _ = build_polygon_from_border(
            delayed.scaled_points,
            label=f"delayed mu_max={delayed.sweep_value:.2f}, t_dev={delayed.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        _, delayed_relative_loss, _ = relative_area_loss(baseline_polygon, delayed_polygon)

        values = sorted(values, key=lambda item: item[0])
        values = [(ca.SCALING_BASELINE_MU_MAX, delayed_relative_loss)] + values
        ax.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            linewidth=1.5,
            label=f"Delay to {ca.t_dev_to_year(t_dev)}",
        )

    ax.set_xlabel("mu_max")
    ax.set_ylabel("Relative loss")
    ax.set_title(
        f"Scaled area loss by mu_max\nbaseline mu_max={ca.SCALING_BASELINE_MU_MAX:.2f}, "
        f"year {ca.t_dev_to_year(ca.SCALING_BASELINE_T_DEV)}, no mu_incr constraint"
    )
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = ca.OUTPUT_DIR / "compensation_mu_max_no_constraint_relative_loss.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def run_mu_max_analysis() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("mu_max no-constraint analysis requires shapely.")
    compare_values = compensation_mu_values()
    rows = ca.compute_mu_max_loss_table(ca.BASELINE_MU_MAX, compare_values)
    losses_by_t_dev = ca.compute_mu_max_loss_series(ca.BASELINE_MU_MAX, compare_values)
    xlsx_path = ca.save_loss_table_xlsx(
        rows,
        ca.OUTPUT_DIR / "compensation_mu_max_no_constraint_loss_table.xlsx",
    )
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


def run_mu_12_loss_region_panel() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("mu=1.2 no-constraint summary panel requires shapely.")

    panel_t_devs = [4, 5, 6, 7, 8, 9]
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=150)
    axes = axes.ravel()

    for idx, comp_t_dev in enumerate(panel_t_devs):
        base, comp = load_scaled_standard_delay(
            ca.SCALING_BASELINE_MU_MAX,
            ca.SCALING_BASELINE_T_DEV,
            comp_t_dev,
        )
        base_polygon, _ = build_polygon_from_border(
            base.scaled_points,
            label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        comp_polygon, _ = build_polygon_from_border(
            comp.scaled_points,
            label=f"comp mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
        ca.draw_lost_region_panel(axes[idx], base, comp, annotate_loss=relative_loss)

    fig.suptitle(
        f"Loss regions for delayed mitigation: baseline mu_max={ca.SCALING_BASELINE_MU_MAX:.2f}, "
        f"year {ca.t_dev_to_year(ca.SCALING_BASELINE_T_DEV)}",
        fontsize=24,
    )
    plt.tight_layout()
    out_path = ca.OUTPUT_DIR / "compensation_mu_max_no_constraint_loss_region_panel_2x3.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved mu_max loss-region 2x3 panel: {out_path}")


def run_delay_compensation_panels() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("delay compensation panels require shapely.")

    baseline, _ = load_scaled_standard_delay(
        ca.SCALING_BASELINE_MU_MAX,
        ca.SCALING_BASELINE_T_DEV,
        ca.SCALING_BASELINE_T_DEV,
    )
    panel_t_devs = [6, 7, 8]
    panel_mu_values = compensation_mu_values()

    for delay_t_dev in panel_t_devs:
        _, delayed = load_scaled_standard_delay(
            ca.SCALING_BASELINE_MU_MAX,
            ca.SCALING_BASELINE_T_DEV,
            delay_t_dev,
        )

        fig, axes = plt.subplots(3, 3, figsize=(24, 18), dpi=150)
        axes = axes.ravel()

        for idx, comp_mu_max in enumerate(panel_mu_values):
            _, compensated = load_scaled_mu_max_comparison(
                ca.SCALING_BASELINE_MU_MAX,
                ca.SCALING_BASELINE_T_DEV,
                comp_mu_max,
                delay_t_dev,
            )
            draw_delay_compensation_panel(
                axes[idx],
                baseline,
                delayed,
                compensated,
                show_title=False,
                annotation_xy=(0.5, 0.5),
                annotation_ha="center",
                annotation_va="center",
            )

        plt.tight_layout()
        fig.subplots_adjust(top=0.97, wspace=0.20, hspace=0.22)
        out_path = ca.OUTPUT_DIR / f"compensation_mu_max_no_constraint_delay_panel_delay{delay_t_dev}_3x3.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved mu_max delay-compensation 3x3 panel: {out_path}")


def run_mu_12_tdev_series() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("mu=1.2 t_dev series requires shapely.")

    available_t_devs = sorted(set(ca.TARGET_T_DEVS + [ca.SCALING_BASELINE_T_DEV]))
    for comp_t_dev in available_t_devs:
        try:
            base, comp = load_scaled_standard_delay(
                ca.SCALING_BASELINE_MU_MAX,
                ca.SCALING_BASELINE_T_DEV,
                comp_t_dev,
            )
        except FileNotFoundError:
            continue

        lost_region_path, debug_stats = ca.plot_lost_region(base, comp, annotate_loss=0.0)
        base_polygon, _ = build_polygon_from_border(
            base.scaled_points,
            label=f"baseline mu_max={base.sweep_value:.2f}, t_dev={base.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        comp_polygon, _ = build_polygon_from_border(
            comp.scaled_points,
            label=f"delayed mu_max={comp.sweep_value:.2f}, t_dev={comp.t_dev}",
            repair_invalid=ca.REPAIR_INVALID_POLYGONS,
        )
        _, relative_loss, _ = relative_area_loss(base_polygon, comp_polygon)
        final_path, _ = ca.plot_lost_region(base, comp, annotate_loss=relative_loss)
        print(
            "Saved mu=1.2 t_dev comparison: "
            f"baseline t_dev={ca.SCALING_BASELINE_T_DEV} vs comp t_dev={comp_t_dev} -> {final_path}"
        )
        if final_path != lost_region_path:
            print(f"Updated preliminary loss-region plot: {lost_region_path}")
        print(
            f"  base_area={debug_stats['base_area_scaled']:.6f}, "
            f"comp_area={debug_stats['comp_area_scaled']:.6f}, "
            f"lost_area={debug_stats['lost_area_scaled']:.6f}, "
            f"relative_loss={relative_loss:.6f}"
        )


def run_delay_compensation_example() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("delay compensation example requires shapely.")

    baseline, delayed = load_scaled_standard_delay(
        ca.SCALING_BASELINE_MU_MAX,
        ca.SCALING_BASELINE_T_DEV,
        ca.DEMO_T_DEV,
    )
    _, compensated = load_scaled_mu_max_comparison(
        ca.SCALING_BASELINE_MU_MAX,
        ca.SCALING_BASELINE_T_DEV,
        ca.DEMO_COMPARED_MU_MAX,
        ca.DEMO_T_DEV,
    )
    out_path, stats = ca.plot_delay_compensation_example(baseline, delayed, compensated)
    print(f"Saved delay-compensation example plot: {out_path}")
    print(
        f"  delay_loss={stats['delayed_relative_loss']:.6f}, "
        f"after_compensation={stats['compensated_relative_loss']:.6f}, "
        f"recovered_loss={stats['recovered_loss_area_scaled']:.6f}"
    )


def run_delay_compensation_series() -> None:
    if not ca.SHAPELY_AVAILABLE:
        raise RuntimeError("delay compensation series requires shapely.")

    for comp_mu_max in compensation_mu_values():
        for comp_t_dev in ca.TARGET_T_DEVS:
            try:
                baseline, delayed = load_scaled_standard_delay(
                    ca.SCALING_BASELINE_MU_MAX,
                    ca.SCALING_BASELINE_T_DEV,
                    comp_t_dev,
                )
                _, compensated = load_scaled_mu_max_comparison(
                    ca.SCALING_BASELINE_MU_MAX,
                    ca.SCALING_BASELINE_T_DEV,
                    comp_mu_max,
                    comp_t_dev,
                )
            except FileNotFoundError:
                continue

            out_path, stats = ca.plot_delay_compensation_example(baseline, delayed, compensated)
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


ca.load_fixed_scaling_bounds = load_fixed_scaling_bounds
ca.available_t_devs_for_mu = available_t_devs_for_mu
ca.common_scaled_axis_limits = common_scaled_axis_limits
ca.common_raw_axis_limits = common_raw_axis_limits
ca.load_scaled_mu_max_pair = load_scaled_mu_max_pair
ca.load_scaled_mu_max_comparison = load_scaled_mu_max_comparison
ca._scenario_role_label = _scenario_role_label
ca.draw_delay_compensation_panel = draw_delay_compensation_panel
ca.plot_mu_max_loss_series = plot_mu_max_loss_series
ca.run_mu_max_analysis = run_mu_max_analysis
ca.run_mu_12_tdev_series = run_mu_12_tdev_series
ca.run_mu_12_loss_region_panel = run_mu_12_loss_region_panel
ca.run_delay_compensation_example = run_delay_compensation_example
ca.run_delay_compensation_series = run_delay_compensation_series
ca.run_delay_compensation_panels = run_delay_compensation_panels


def main() -> None:
    if ca.RUN_DEMO:
        ca.run_demo()
    if ca.RUN_MU_MAX_ANALYSIS:
        ca.run_mu_max_analysis()
    if ca.RUN_MU_12_TDEV_SERIES:
        ca.run_mu_12_tdev_series()
        ca.run_mu_12_loss_region_panel()
    if ca.RUN_MU_MAX_LOSS_REGIONS:
        ca.run_mu_max_loss_regions()
    if ca.RUN_DELAY_COMPENSATION_EXAMPLE:
        ca.run_delay_compensation_example()
    if ca.RUN_DELAY_COMPENSATION_SERIES:
        ca.run_delay_compensation_series()
        ca.run_delay_compensation_panels()


if __name__ == "__main__":
    main()
