from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import scenario_config as sc


# Enable any subset of scenarios for plotting.
PLOT_CHANGE_MU_MAX = False
PLOT_CHANGE_T_TAR = False
PLOT_CHANGE_MU_MAX_OPTIMAL_MU = False
PLOT_CHANGE_MU_INCR = True
PLOT_WITH_MARKERS = False  # Set False for line-only plots (no markers).

OUTPUT_DIR = Path("plots_png")
OUTPUT_DIR.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 10,
        "lines.linewidth": 1.2,
        "lines.markersize": 6,
    }
)


COLORS = ["red", "blue", "green", "orange", "brown", "cyan", "magenta"]
MARKERS = ["o", "<", ">", "^", "s", "D", "P"]


def _configured_t_dev_values():
    t_dev_values = set()
    scenarios = [
        sc.SCENARIO_CHANGE_MU_MAX,
        sc.SCENARIO_CHANGE_T_TAR,
        sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU,
        sc.SCENARIO_CHANGE_MU_INCR,
    ]
    for scenario_name in scenarios:
        baseline = sc.scenario_baseline(scenario_name)
        for sweep_value in sc.scenario_sweep_values(scenario_name):
            t_dev_values.update(sc.t_dev_runs(sweep_value, baseline))
    if not t_dev_values:
        raise ValueError("No t_dev values found from scenario_config.")
    return sorted(t_dev_values)


def _build_style_by_tdev():
    style = {}
    for idx, t_dev in enumerate(_configured_t_dev_values()):
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        style[t_dev] = (color, marker)
    return style


STYLE_BY_TDEV = _build_style_by_tdev()


def _scenario_label(scenario_name, sweep_value):
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX:
        return f"mu_max={sweep_value:.2f}", f"RS for mu_max={sweep_value:.2f}"
    if scenario_name == sc.SCENARIO_CHANGE_T_TAR:
        year = 2015 + 5 * int(round(sweep_value))
        return f"t_tar={year}", f"RS for t_tar={year}"
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return f"mu_max={sweep_value:.2f}, mu=mu_opt", f"RS for mu_max={sweep_value:.2f} (mu=mu_opt)"
    if scenario_name == sc.SCENARIO_CHANGE_MU_INCR:
        if sweep_value == sc.MU_INCR_NO_CONSTRAINT:
            return "mu_incr=no constraint", "RS for mu_incr=no constraint"
        return f"mu_incr={float(sweep_value):.2f}", f"RS for mu_incr={float(sweep_value):.2f}"
    raise ValueError(f"Unknown scenario: {scenario_name}")


def _format_sweep_value(sweep_value):
    if isinstance(sweep_value, str):
        return sweep_value.replace(" ", "_")
    return f"{float(sweep_value):.2f}"


def _load_points(path):
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _plot_series(ax, points, color, marker, label):
    if PLOT_WITH_MARKERS:
        ax.plot(
            points[:, 0],
            points[:, 1],
            linestyle="None",
            marker=marker,
            color=color,
            label=label,
        )
        ax.add_patch(Polygon(points, fill=False, edgecolor=color, linewidth=1.2))
    else:
        x_closed = np.append(points[:, 0], points[0, 0])
        y_closed = np.append(points[:, 1], points[0, 1])
        ax.plot(
            x_closed,
            y_closed,
            linestyle="-",
            marker=None,
            color=color,
            linewidth=1.0,
            label=label,
        )


def _plot_one_scenario(scenario_name):
    baseline = sc.scenario_baseline(scenario_name)
    sweep_values = sc.scenario_sweep_values(scenario_name)
    reference_candidates = sorted(set(sc.t_dev_runs(baseline, baseline)))
    reference_t_dev = reference_candidates[0]
    reference_file = Path(sc.data_filename(scenario_name, baseline, reference_t_dev))
    reference_points = None

    if reference_file.exists():
        reference_points = _load_points(reference_file)
    else:
        print(f"Reference file not found: {reference_file}")

    for sweep_value in sweep_values:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        all_points = []

        if reference_points is not None:
            ref_color, ref_marker = STYLE_BY_TDEV[reference_t_dev]
            _plot_series(
                ax,
                reference_points,
                ref_color,
                ref_marker,
                f"Year {2015 + 5 * reference_t_dev} (reference)",
            )
            all_points.append(reference_points)

        for t_dev in sc.t_dev_runs(sweep_value, baseline):
            if reference_points is not None and t_dev == reference_t_dev:
                continue
            path = Path(sc.data_filename(scenario_name, sweep_value, t_dev))
            if not path.exists():
                print(f"Missing data file: {path}")
                continue

            points = _load_points(path)
            color, marker = STYLE_BY_TDEV[t_dev]
            _plot_series(ax, points, color, marker, f"Year {2015 + 5 * t_dev}")
            all_points.append(points)

        if not all_points:
            plt.close(fig)
            print(f"No plottable points for scenario={scenario_name}, sweep={_format_sweep_value(sweep_value)}")
            continue

        merged = np.vstack(all_points)
        x_min, x_max = merged[:, 0].min(), merged[:, 0].max()
        y_min, y_max = merged[:, 1].min(), merged[:, 1].max()
        x_pad = max(5.0, 0.05 * (x_max - x_min))
        y_pad = max(0.05, 0.08 * (y_max - y_min))

        ax.set_xlabel("Q in 2100")
        ax.set_ylabel("Delta T in 2100 (C)")
        sweep_label, title = _scenario_label(scenario_name, sweep_value)
        ax.set_title(title)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([0.0, y_max + y_pad])
        ax.grid(True)
        ax.legend(loc="center", frameon=True)
        plt.tight_layout()

        out_name = f"{scenario_name}_{_format_sweep_value(sweep_value)}.png"
        out_path = OUTPUT_DIR / out_name
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path} ({sweep_label})")


if __name__ == "__main__":
    selected = []
    if PLOT_CHANGE_MU_MAX:
        selected.append(sc.SCENARIO_CHANGE_MU_MAX)
    if PLOT_CHANGE_T_TAR:
        selected.append(sc.SCENARIO_CHANGE_T_TAR)
    if PLOT_CHANGE_MU_MAX_OPTIMAL_MU:
        selected.append(sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU)
    if PLOT_CHANGE_MU_INCR:
        selected.append(sc.SCENARIO_CHANGE_MU_INCR)

    if not selected:
        raise ValueError("Enable at least one scenario flag in plotter.py.")

    for scenario in selected:
        _plot_one_scenario(scenario)

    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")
