"""Compute and plot the signed relative opportunity-space area gap (Figure 5).

For each displayed scenario i, the metric is

    G_i = (S_base - S_i) / S_base,

where S_base is the target-year-specific baseline polygon area and S_i is the
scenario polygon area. Positive values indicate a smaller opportunity space
than baseline; negative values indicate a larger opportunity space.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.ticker import PercentFormatter

from make_scaled_area_loss_heatmaps import (
    BASELINE_MU_INCR,
    BASELINE_MU_MAX,
    MU_INCR_VALUES,
    MU_MAX_VALUES,
    OUTPUT_DIR,
    TARGETS,
    T_DEVS,
    border_path,
    cell_edges,
    load_points,
    polygon,
    scaled,
)


BASELINE_T_DEV = 1
FIGURE_PATH = OUTPUT_DIR / "relative_area_gap_heatmaps_3x5.png"
ZERO_BOUNDARY_FIGURE_PATH = OUTPUT_DIR / "relative_area_gap_heatmaps_3x5_zero_boundary.png"
TABLE_PATH = OUTPUT_DIR / "relative_area_gap_heatmaps_3x5.csv"
DIAGNOSTIC_PATH = OUTPUT_DIR / "figure4_figure5_area_metric_diagnostic.csv"
SUMMARY_PATH = OUTPUT_DIR / "relative_area_gap_panel_summary.csv"


def relative_area_gap_matrix(target_t: int, target_year: int, t_dev: int) -> np.ndarray:
    """Return (S_base - S_i) / S_base on the exact 7-by-9 grid."""
    baseline_path = border_path(
        BASELINE_MU_MAX,
        BASELINE_MU_INCR,
        BASELINE_T_DEV,
        target_t,
    )
    baseline_raw = load_points(baseline_path)
    bounds = (
        float(baseline_raw[:, 0].min()),
        float(baseline_raw[:, 0].max()),
        float(baseline_raw[:, 1].min()),
        float(baseline_raw[:, 1].max()),
    )
    baseline = polygon(
        scaled(baseline_raw, bounds),
        f"baseline {target_year}",
    )
    baseline_area = float(baseline.area)
    if baseline_area <= 0.0:
        raise ValueError(f"Baseline polygon for {target_year} has non-positive area.")

    matrix = np.full((len(MU_INCR_VALUES), len(MU_MAX_VALUES)), np.nan)
    for row, mu_incr in enumerate(MU_INCR_VALUES):
        for col, mu_max in enumerate(MU_MAX_VALUES):
            path = border_path(mu_max, mu_incr, t_dev, target_t)
            scenario = polygon(
                scaled(load_points(path), bounds),
                path.name,
            )
            matrix[row, col] = (baseline_area - float(scenario.area)) / baseline_area
    return matrix


def discrete_sign_change_segments(
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Return cell-edge segments separating adjacent values of opposite sign."""
    n_rows, n_cols = matrix.shape
    segments: list[list[tuple[float, float]]] = []

    for row in range(n_rows):
        for boundary_col in range(1, n_cols):
            left = matrix[row, boundary_col - 1]
            right = matrix[row, boundary_col]
            if left * right < 0.0:
                segments.append(
                    [
                        (x_edges[boundary_col], y_edges[row]),
                        (x_edges[boundary_col], y_edges[row + 1]),
                    ]
                )

    for boundary_row in range(1, n_rows):
        for col in range(n_cols):
            below = matrix[boundary_row - 1, col]
            above = matrix[boundary_row, col]
            if below * above < 0.0:
                segments.append(
                    [
                        (x_edges[col], y_edges[boundary_row]),
                        (x_edges[col + 1], y_edges[boundary_row]),
                    ]
                )

    return segments


def plot_publication_area_gap_heatmaps(
    matrices: dict[tuple[int, int], np.ndarray],
    output: Path,
    *,
    show_zero_boundary: bool = False,
    x_tick_label_step: int = 1,
) -> colors.TwoSlopeNorm:
    """Render Figure 5 as discrete cells with a shared zero-centered scale."""
    expected_shape = (len(MU_INCR_VALUES), len(MU_MAX_VALUES))
    if x_tick_label_step < 1:
        raise ValueError("x_tick_label_step must be at least 1.")
    for key, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"Matrix {key} has shape {matrix.shape}; expected {expected_shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Matrix {key} contains a non-finite grid value.")

    max_absolute_gap = max(
        float(np.max(np.abs(matrix)))
        for matrix in matrices.values()
    )
    color_limit = max(max_absolute_gap, 1e-12)
    norm = colors.TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )
    x_edges = cell_edges(MU_MAX_VALUES)
    y_edges = cell_edges(MU_INCR_VALUES)

    fig, axes = plt.subplots(
        len(T_DEVS),
        len(TARGETS),
        figsize=(8.2, 10.2),
        dpi=300,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.23,
        right=0.86,
        bottom=0.11,
        top=0.91,
        wspace=0.08,
        hspace=0.10,
    )

    mesh = None
    for col, (target_t, target_year) in enumerate(TARGETS):
        for row, t_dev in enumerate(T_DEVS):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrices[(target_t, t_dev)],
                shading="flat",
                cmap="RdBu_r",
                norm=norm,
                edgecolors="none",
                antialiased=False,
                rasterized=True,
            )

            if show_zero_boundary:
                segments = discrete_sign_change_segments(
                    matrices[(target_t, t_dev)],
                    x_edges,
                    y_edges,
                )
                if segments:
                    ax.add_collection(
                        LineCollection(
                            segments,
                            colors="0.25",
                            linewidths=0.7,
                            zorder=3,
                        )
                    )

            ax.scatter(
                [BASELINE_MU_MAX],
                [BASELINE_MU_INCR],
                s=30,
                facecolors="none",
                edgecolors="0.12",
                linewidths=0.8,
                zorder=5,
            )

            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.set_xticks(MU_MAX_VALUES)
            ax.set_yticks(MU_INCR_VALUES)
            ax.tick_params(
                axis="both",
                labelsize=7,
                length=2.2,
                width=0.6,
                pad=2,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.label_outer()
            if row == 0:
                ax.set_title(str(target_year), fontsize=10, pad=6)

    x_tick_labels = [
        f"{value:.1f}" if index % x_tick_label_step == 0 else ""
        for index, value in enumerate(MU_MAX_VALUES)
    ]
    for ax in axes[-1, :]:
        ax.set_xticklabels(x_tick_labels)
        plt.setp(
            ax.get_xticklabels(),
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )

    for row, t_dev in enumerate(T_DEVS):
        position = axes[row, 0].get_position()
        fig.text(
            0.075,
            0.5 * (position.y0 + position.y1),
            str(2015 + 5 * t_dev),
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.text(0.545, 0.965, "Terminal year", ha="center", va="center", fontsize=11)
    fig.text(
        0.022,
        0.51,
        "Mitigation delayed until",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
    )
    fig.text(
        0.545,
        0.025,
        r"Maximum admissible emission-reduction rate, $\mu_{\max}$",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.145,
        0.51,
        r"Maximum growth factor, $\mu_{\mathrm{incr}}$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10,
    )

    if mesh is not None:
        colorbar_ax = fig.add_axes([0.89, 0.18, 0.022, 0.65])
        colorbar_ticks = np.array(
            [-color_limit, -0.5 * color_limit, 0.0, 0.5 * color_limit, color_limit]
        )
        colorbar = fig.colorbar(mesh, cax=colorbar_ax, ticks=colorbar_ticks)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)
        colorbar.outline.set_linewidth(0.6)
        colorbar.set_label("Relative area gap", fontsize=10, labelpad=8)

    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return norm


def save_area_gap_table(
    matrices: dict[tuple[int, int], np.ndarray],
    output: Path,
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_year",
                "delay_year",
                "mu_incr",
                *[f"mu_max={value:.1f}" for value in MU_MAX_VALUES],
            ]
        )
        for target_t, target_year in TARGETS:
            for t_dev in T_DEVS:
                matrix = matrices[(target_t, t_dev)]
                for row, mu_incr in enumerate(MU_INCR_VALUES):
                    writer.writerow(
                        [target_year, 2015 + 5 * t_dev, mu_incr, *matrix[row]]
                    )


def compute_area_metric_diagnostics(
    matrices: dict[tuple[int, int], np.ndarray],
) -> list[dict[str, float | int | bool]]:
    """Compare Figure 4 and 5 metrics and verify their geometric identity."""
    rows: list[dict[str, float | int | bool]] = []
    for target_t, target_year in TARGETS:
        baseline_raw = load_points(
            border_path(
                BASELINE_MU_MAX,
                BASELINE_MU_INCR,
                BASELINE_T_DEV,
                target_t,
            )
        )
        bounds = (
            float(baseline_raw[:, 0].min()),
            float(baseline_raw[:, 0].max()),
            float(baseline_raw[:, 1].min()),
            float(baseline_raw[:, 1].max()),
        )
        baseline = polygon(scaled(baseline_raw, bounds), f"baseline {target_year}")
        baseline_area = float(baseline.area)

        for t_dev in T_DEVS:
            area_gap = matrices[(target_t, t_dev)]
            for row_index, mu_incr in enumerate(MU_INCR_VALUES):
                for col_index, mu_max in enumerate(MU_MAX_VALUES):
                    path = border_path(mu_max, mu_incr, t_dev, target_t)
                    scenario = polygon(
                        scaled(load_points(path), bounds),
                        path.name,
                    )
                    uncovered_loss = float(baseline.difference(scenario).area) / baseline_area
                    relative_area_gap = float(area_gap[row_index, col_index])
                    gap_difference = uncovered_loss - relative_area_gap
                    new_outside_baseline = float(scenario.difference(baseline).area) / baseline_area
                    residual = gap_difference - new_outside_baseline
                    rows.append(
                        {
                            "target_year": int(target_year),
                            "delay_year": int(2015 + 5 * t_dev),
                            "mu_max": float(mu_max),
                            "mu_incr": float(mu_incr),
                            "relative_uncovered_loss": uncovered_loss,
                            "relative_area_gap": relative_area_gap,
                            "gap_LU_minus_LA": gap_difference,
                            "new_opportunity_outside_baseline": new_outside_baseline,
                            "identity_residual": residual,
                            "area_expands_but_uncovered_loss_positive": bool(
                                relative_area_gap < 0.0 and uncovered_loss > 0.0
                            ),
                        }
                    )

    max_residual = max(abs(float(row["identity_residual"])) for row in rows)
    if max_residual > 1e-10:
        raise AssertionError(
            "L_U - L_A identity failed: "
            f"maximum absolute residual is {max_residual:.3e}."
        )
    return rows


def panel_summary_rows(
    matrices: dict[tuple[int, int], np.ndarray],
) -> list[dict[str, float | int | bool]]:
    benchmark_row = MU_INCR_VALUES.index(BASELINE_MU_INCR)
    benchmark_col = MU_MAX_VALUES.index(BASELINE_MU_MAX)
    rows: list[dict[str, float | int | bool]] = []
    for target_t, target_year in TARGETS:
        for t_dev in T_DEVS:
            matrix = matrices[(target_t, t_dev)]
            minimum = float(matrix.min())
            maximum = float(matrix.max())
            rows.append(
                {
                    "target_year": int(target_year),
                    "delay_year": int(2015 + 5 * t_dev),
                    "benchmark_mu_max": float(BASELINE_MU_MAX),
                    "benchmark_mu_incr": float(BASELINE_MU_INCR),
                    "benchmark_relative_area_gap": float(
                        matrix[benchmark_row, benchmark_col]
                    ),
                    "minimum_relative_area_gap": minimum,
                    "maximum_relative_area_gap": maximum,
                    "sign_changes_in_panel": bool(minimum < 0.0 < maximum),
                }
            )
    return rows


def save_dict_rows(
    rows: list[dict[str, float | int | bool]],
    output: Path,
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {output}.")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = {
        (target_t, t_dev): relative_area_gap_matrix(target_t, target_year, t_dev)
        for target_t, target_year in TARGETS
        for t_dev in T_DEVS
    }
    norm = plot_publication_area_gap_heatmaps(
        matrices,
        FIGURE_PATH,
        show_zero_boundary=False,
        x_tick_label_step=2,
    )
    plot_publication_area_gap_heatmaps(
        matrices,
        ZERO_BOUNDARY_FIGURE_PATH,
        show_zero_boundary=True,
        x_tick_label_step=2,
    )
    save_area_gap_table(matrices, TABLE_PATH)
    diagnostic_rows = compute_area_metric_diagnostics(matrices)
    summary_rows = panel_summary_rows(matrices)
    save_dict_rows(diagnostic_rows, DIAGNOSTIC_PATH)
    save_dict_rows(summary_rows, SUMMARY_PATH)

    minimum = min(float(matrix.min()) for matrix in matrices.values())
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    gap_minimum = min(float(row["gap_LU_minus_LA"]) for row in diagnostic_rows)
    gap_maximum = max(float(row["gap_LU_minus_LA"]) for row in diagnostic_rows)
    max_residual = max(abs(float(row["identity_residual"])) for row in diagnostic_rows)
    expanding_with_loss = [
        row
        for row in diagnostic_rows
        if bool(row["area_expands_but_uncovered_loss_positive"])
    ]
    representatives: list[dict[str, float | int | bool]] = []
    for _, target_year in TARGETS:
        target_rows = [
            row
            for row in expanding_with_loss
            if int(row["target_year"]) == target_year
        ]
        largest = max(target_rows, key=lambda item: float(item["gap_LU_minus_LA"]))
        median_gap = float(np.median([float(row["gap_LU_minus_LA"]) for row in target_rows]))
        moderate = min(
            target_rows,
            key=lambda item: abs(float(item["gap_LU_minus_LA"]) - median_gap),
        )
        representatives.extend([largest, moderate])

    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {ZERO_BOUNDARY_FIGURE_PATH}")
    print(f"Saved: {TABLE_PATH}")
    print(f"Saved: {DIAGNOSTIC_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Relative area-gap range: {minimum:.6f} .. {maximum:.6f}")
    print(f"Shared symmetric color scale: {norm.vmin:.6f} .. {norm.vmax:.6f}")
    print(f"G = L_U - L_A range: {gap_minimum:.6f} .. {gap_maximum:.6f}")
    print(f"Maximum identity residual: {max_residual:.3e}")
    print(
        "Scenarios with L_A < 0 and L_U > 0: "
        f"{len(expanding_with_loss)} / {len(diagnostic_rows)}"
    )
    print("Representative expanding scenarios with uncovered loss:")
    for row in representatives:
        print(
            "  "
            f"terminal={int(row['target_year'])}, "
            f"delay={int(row['delay_year'])}, "
            f"mu_max={float(row['mu_max']):.1f}, "
            f"mu_incr={float(row['mu_incr']):.1f}, "
            f"L_U={float(row['relative_uncovered_loss']):.4f}, "
            f"L_A={float(row['relative_area_gap']):.4f}, "
            f"G={float(row['gap_LU_minus_LA']):.4f}"
        )


if __name__ == "__main__":
    main()
