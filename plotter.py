from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import scenario_config as sc


# Enable any subset of scenarios for plotting.
PLOT_CHANGE_MU_MAX = True
PLOT_CHANGE_T_TAR = False
PLOT_CHANGE_MU_MAX_OPTIMAL_MU = False

OUTPUT_DIR = Path("plots_png")
OUTPUT_DIR.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


COLORS = ["red", "blue", "green", "orange", "brown", "cyan", "magenta"]
MARKERS = ["o", "<", ">", "^", "s", "D", "P"]
STYLE_BY_TDEV = {t_dev: (COLORS[t_dev - 1], MARKERS[t_dev - 1]) for t_dev in range(1, 8)}


def _scenario_label(scenario_name, sweep_value):
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX:
        return f"mu_max={sweep_value:.2f}", f"RS for mu_max={sweep_value:.2f}"
    if scenario_name == sc.SCENARIO_CHANGE_T_TAR:
        year = 2015 + 5 * int(round(sweep_value))
        return f"t_tar={year}", f"RS for t_tar={year}"
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return f"mu_max={sweep_value:.2f}, mu=mu_opt", f"RS for mu_max={sweep_value:.2f} (mu=mu_opt)"
    raise ValueError(f"Unknown scenario: {scenario_name}")


def _load_points(path):
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _plot_one_scenario(scenario_name):
    baseline = sc.scenario_baseline(scenario_name)
    sweep_values = sc.scenario_sweep_values(scenario_name)
    reference_t_dev = 1
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
            ax.plot(
                reference_points[:, 0],
                reference_points[:, 1],
                linestyle="None",
                marker=ref_marker,
                color=ref_color,
                label=f"Year {2015 + 5 * reference_t_dev} (reference)",
            )
            ax.add_patch(Polygon(reference_points, fill=False, edgecolor=ref_color, linewidth=1.2))
            all_points.append(reference_points)

        for t_dev in sc.t_dev_runs(sweep_value, baseline):
            path = Path(sc.data_filename(scenario_name, sweep_value, t_dev))
            if not path.exists():
                print(f"Missing data file: {path}")
                continue

            points = _load_points(path)
            color, marker = STYLE_BY_TDEV[t_dev]
            ax.plot(
                points[:, 0],
                points[:, 1],
                linestyle="None",
                marker=marker,
                color=color,
                label=f"Year {2015 + 5 * t_dev}",
            )
            ax.add_patch(Polygon(points, fill=False, edgecolor=color, linewidth=1.2))
            all_points.append(points)

        if not all_points:
            plt.close(fig)
            print(f"No plottable points for scenario={scenario_name}, sweep={sweep_value:.2f}")
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
        ax.set_ylim([max(0.0, y_min - y_pad), y_max + y_pad])
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        out_name = f"{scenario_name}_{sweep_value:.2f}.png"
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

    if not selected:
        raise ValueError("Enable at least one scenario flag in plotter.py.")

    for scenario in selected:
        _plot_one_scenario(scenario)

    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")
