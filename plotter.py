from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import scenario_config as sc
from scaling_utils import compute_global_scaling_bounds, normalize_points


# Enable any subset of scenarios for plotting.
PLOT_CHANGE_MU_MAX = True
PLOT_CHANGE_MU_MAX_NO_CONSTRAINT = False
PLOT_CHANGE_MU_MAX_AND_MU_INCR = False
PLOT_CHANGE_T_TAR = False
PLOT_CHANGE_MU_MAX_OPTIMAL_MU = False
PLOT_CHANGE_MU_INCR = False
PLOT_WITH_MARKERS = False  # Set False for line-only plots (no markers).
PLOT_IN_SCALED_COORDS = False  # Uses one common scale across all series in a figure.

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

_PLOT_SCENARIO_ENV_NAME = "NCC_PLOT_SCENARIO"
_PLOT_FLAG_BY_NAME = {
    sc.SCENARIO_CHANGE_MU_MAX: "PLOT_CHANGE_MU_MAX",
    sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT: "PLOT_CHANGE_MU_MAX_NO_CONSTRAINT",
    sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR: "PLOT_CHANGE_MU_MAX_AND_MU_INCR",
    sc.SCENARIO_CHANGE_T_TAR: "PLOT_CHANGE_T_TAR",
    sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU: "PLOT_CHANGE_MU_MAX_OPTIMAL_MU",
    sc.SCENARIO_CHANGE_MU_INCR: "PLOT_CHANGE_MU_INCR",
}


def _apply_plot_scenario_override_from_env():
    override = os.environ.get(_PLOT_SCENARIO_ENV_NAME)
    if not override:
        return
    if override not in _PLOT_FLAG_BY_NAME:
        allowed = ", ".join(sorted(_PLOT_FLAG_BY_NAME))
        raise ValueError(
            f"Unsupported {_PLOT_SCENARIO_ENV_NAME}={override!r}. Allowed values: {allowed}"
        )

    for flag_name in _PLOT_FLAG_BY_NAME.values():
        globals()[flag_name] = False
    globals()[_PLOT_FLAG_BY_NAME[override]] = True


_apply_plot_scenario_override_from_env()


def _configured_t_dev_values():
    t_dev_values = set()
    scenarios = [
        sc.SCENARIO_CHANGE_MU_MAX,
        sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT,
        sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR,
        sc.SCENARIO_CHANGE_T_TAR,
        sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU,
        sc.SCENARIO_CHANGE_MU_INCR,
    ]
    for scenario_name in scenarios:
        for sweep_value in sc.scenario_sweep_values(scenario_name):
            t_dev_values.update(sc.scenario_t_dev_runs(scenario_name, sweep_value))
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
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT:
        return (
            f"mu_max={sweep_value:.2f}, mu_incr=none",
            f"RS for mu_max={sweep_value:.2f} (mu_incr=none)",
        )
    if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR:
        mu_max, mu_incr = sweep_value
        return (
            f"mu_max={mu_max:.2f}, mu_incr={mu_incr:.2f}",
            f"RS for mu_max={mu_max:.2f}, mu_incr={mu_incr:.2f}",
        )
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
    if isinstance(sweep_value, (tuple, list)) and len(sweep_value) == 2:
        return f"mu{float(sweep_value[0]):.2f}_muincr{float(sweep_value[1]):.2f}"
    if isinstance(sweep_value, str):
        return sweep_value.replace(" ", "_")
    return f"{float(sweep_value):.2f}"


def _load_points_with_legacy_fallback(path: Path, legacy_path: Path | None = None):
    if path.exists():
        return _load_points(path)
    if legacy_path is not None and legacy_path.exists():
        print(f"Using legacy file name: {legacy_path}")
        return _load_points(legacy_path)
    raise FileNotFoundError(path)


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
    t_tar_values = sc.scenario_t_tar_values(scenario_name)
    reference_candidates = sorted(set(sc.scenario_t_dev_runs(scenario_name, baseline)))
    reference_t_dev = reference_candidates[0]
    for t_tar in t_tar_values:
        reference_file = Path(sc.data_filename(scenario_name, baseline, reference_t_dev, t_tar=t_tar))
        reference_points = None

        legacy_reference = None
        if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR and int(round(t_tar)) == sc.DEFAULT_T_TAR:
            legacy_reference = Path(
                f"plots_data/NCC_mu{baseline[0]:.2f}_muincr{baseline[1]:.2f}_tdev{reference_t_dev}.csv"
            )

        try:
            reference_points = _load_points_with_legacy_fallback(reference_file, legacy_reference)
        except FileNotFoundError:
            print(f"Reference file not found: {reference_file}")

        for sweep_value in sweep_values:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
            series_to_plot = []

            if reference_points is not None:
                ref_color, ref_marker = STYLE_BY_TDEV[reference_t_dev]
                series_to_plot.append(
                    (
                        f"Year {2015 + 5 * reference_t_dev} (reference)",
                        reference_points,
                        ref_color,
                        ref_marker,
                    )
                )

            for t_dev in sc.scenario_t_dev_runs(scenario_name, sweep_value):
                if reference_points is not None and t_dev == reference_t_dev:
                    continue
                path = Path(sc.data_filename(scenario_name, sweep_value, t_dev, t_tar=t_tar))
                legacy_path = None
                if scenario_name == sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR and int(round(t_tar)) == sc.DEFAULT_T_TAR:
                    mu_max, mu_incr = sweep_value
                    legacy_path = Path(
                        f"plots_data/NCC_mu{mu_max:.2f}_muincr{mu_incr:.2f}_tdev{t_dev}.csv"
                    )

                try:
                    points = _load_points_with_legacy_fallback(path, legacy_path)
                except FileNotFoundError:
                    print(f"Missing data file: {path}")
                    continue

                color, marker = STYLE_BY_TDEV[t_dev]
                series_to_plot.append((f"Year {2015 + 5 * t_dev}", points, color, marker))

            if not series_to_plot:
                plt.close(fig)
                print(
                    f"No plottable points for scenario={scenario_name}, "
                    f"sweep={_format_sweep_value(sweep_value)}, t_tar={int(round(t_tar))}"
                )
                continue

            raw_all_points = [points for _, points, _, _ in series_to_plot]
            if PLOT_IN_SCALED_COORDS:
                scaling_bounds = compute_global_scaling_bounds(raw_all_points)
                plotted_series = [
                    (label, normalize_points(points, scaling_bounds), color, marker)
                    for label, points, color, marker in series_to_plot
                ]
            else:
                plotted_series = series_to_plot

            for label, points, color, marker in plotted_series:
                _plot_series(ax, points, color, marker, label)

            merged = np.vstack([points for _, points, _, _ in plotted_series])
            x_min, x_max = merged[:, 0].min(), merged[:, 0].max()
            y_min, y_max = merged[:, 1].min(), merged[:, 1].max()
            x_pad = max(5.0, 0.05 * (x_max - x_min))
            y_pad = max(0.05, 0.08 * (y_max - y_min))

            target_year = sc.t_index_to_year(t_tar)
            if PLOT_IN_SCALED_COORDS:
                x_pad = max(0.02, 0.05 * (x_max - x_min))
                y_pad = max(0.02, 0.08 * (y_max - y_min))
                ax.set_xlabel("Scaled Q")
                ax.set_ylabel("Scaled Delta T")
            else:
                ax.set_xlabel(f"Q in {target_year}")
                ax.set_ylabel(f"Delta T in {target_year} (C)")
            sweep_label, title = _scenario_label(scenario_name, sweep_value)
            ax.set_title(f"{title}, target year {target_year}")
            ax.set_xlim([x_min - x_pad, x_max + x_pad])
            ax.set_ylim([max(0.0, y_min - y_pad), y_max + y_pad] if not PLOT_IN_SCALED_COORDS else [y_min - y_pad, y_max + y_pad])
            ax.grid(True)
            ax.legend(loc="center", frameon=True)
            plt.tight_layout()

            suffix = "_scaled" if PLOT_IN_SCALED_COORDS else ""
            out_name = (
                f"{scenario_name}_{_format_sweep_value(sweep_value)}_year{target_year}{suffix}.png"
            )
            out_path = OUTPUT_DIR / out_name
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved: {out_path} ({sweep_label}, target year {target_year})")


if __name__ == "__main__":
    selected = []
    if PLOT_CHANGE_MU_MAX:
        selected.append(sc.SCENARIO_CHANGE_MU_MAX)
    if PLOT_CHANGE_MU_MAX_NO_CONSTRAINT:
        selected.append(sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT)
    if PLOT_CHANGE_MU_MAX_AND_MU_INCR:
        selected.append(sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR)
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
