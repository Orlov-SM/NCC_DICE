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
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import scenario_config as sc
from geometry_utils import (
    SHAPELY_AVAILABLE,
    build_polygon_from_border,
    geometry_area,
    relative_area_loss,
)
from scaling_utils import ScalingBounds, normalize_points


OUTPUT_DIR = Path("plots_png") / "compensation_mu_max_and_mu_incr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.8,
    }
)

BASELINE_MU_MAX = sc.DEFAULT_MU_MAX
BASELINE_MU_INCR = sc.DEFAULT_MU_INCR
BASELINE_T_DEV = 1
TARGET_T_TAR = int(round(float(os.environ.get("NCC_COMPENSATION_MU_MAX_AND_MU_INCR_T_TAR", sc.DEFAULT_T_TAR))))
TARGET_YEAR = sc.t_index_to_year(TARGET_T_TAR)
TARGET_T_DEVS = [4, 5, 6, 7, 8, 9]
REPAIR_INVALID_POLYGONS = False


def mu_max_values() -> list[float]:
    sweep_values = sc.scenario_sweep_values(sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR)
    return sorted({float(mu_max) for mu_max, _ in sweep_values})


def mu_incr_values() -> list[float]:
    sweep_values = sc.scenario_sweep_values(sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR)
    return sorted({float(mu_incr) for _, mu_incr in sweep_values})


@dataclass(frozen=True)
class BorderScenario:
    mu_max: float
    mu_incr: float
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


def _legacy_border_path(mu_max: float, mu_incr: float, t_dev: int) -> Path:
    return Path(f"plots_data/NCC_mu{mu_max:.2f}_muincr{mu_incr:.2f}_tdev{t_dev}.csv")


def _border_path(mu_max: float, mu_incr: float, t_dev: int) -> Path:
    return Path(
        sc.data_filename(
            sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR,
            (mu_max, mu_incr),
            t_dev,
            t_tar=TARGET_T_TAR,
        )
    )


def load_border(mu_max: float, mu_incr: float, t_dev: int) -> np.ndarray:
    path = _border_path(mu_max, mu_incr, t_dev)
    if path.exists():
        return _load_points(path)
    if TARGET_T_TAR == sc.DEFAULT_T_TAR:
        legacy_path = _legacy_border_path(mu_max, mu_incr, t_dev)
        if legacy_path.exists():
            return _load_points(legacy_path)
    raise FileNotFoundError(f"Missing 2D-scenario border file: {path}")


def load_fixed_scaling_bounds() -> ScalingBounds:
    baseline_raw = load_border(BASELINE_MU_MAX, BASELINE_MU_INCR, BASELINE_T_DEV)
    return ScalingBounds(
        q_min=float(np.min(baseline_raw[:, 0])),
        q_max=float(np.max(baseline_raw[:, 0])),
        temp_min=float(np.min(baseline_raw[:, 1])),
        temp_max=float(np.max(baseline_raw[:, 1])),
    )


@lru_cache(maxsize=1)
def common_scaled_axis_limits() -> tuple[float, float, float, float]:
    bounds = load_fixed_scaling_bounds()
    all_points = []
    for mu_incr in mu_incr_values():
        for mu_max in mu_max_values():
            t_devs = sc.scenario_t_dev_runs(
                sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR,
                (mu_max, mu_incr),
            )
            for t_dev in t_devs:
                try:
                    raw_points = load_border(mu_max, mu_incr, t_dev)
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


def load_scaled_scenario(mu_max: float, mu_incr: float, t_dev: int) -> BorderScenario:
    bounds = load_fixed_scaling_bounds()
    raw_points = load_border(mu_max, mu_incr, t_dev)
    return BorderScenario(
        mu_max=mu_max,
        mu_incr=mu_incr,
        t_dev=t_dev,
        raw_points=raw_points,
        scaled_points=normalize_points(raw_points, bounds),
    )


def compute_loss_rows() -> list[dict[str, float | int]]:
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("2D compensation analysis requires shapely.")

    baseline = load_scaled_scenario(BASELINE_MU_MAX, BASELINE_MU_INCR, BASELINE_T_DEV)
    baseline_polygon, _ = build_polygon_from_border(
        baseline.scaled_points,
        label=(
            f"baseline mu_max={BASELINE_MU_MAX:.2f}, "
            f"mu_incr={BASELINE_MU_INCR:.2f}, t_dev={BASELINE_T_DEV}"
        ),
        repair_invalid=REPAIR_INVALID_POLYGONS,
    )
    baseline_area = geometry_area(baseline_polygon)
    bounds = load_fixed_scaling_bounds()

    rows: list[dict[str, float | int]] = []
    for t_dev in TARGET_T_DEVS:
        for mu_incr in mu_incr_values():
            for mu_max in mu_max_values():
                try:
                    comp = load_scaled_scenario(mu_max, mu_incr, t_dev)
                except FileNotFoundError:
                    continue
                comp_polygon, _ = build_polygon_from_border(
                    comp.scaled_points,
                    label=(
                        f"comp mu_max={mu_max:.2f}, "
                        f"mu_incr={mu_incr:.2f}, t_dev={t_dev}"
                    ),
                    repair_invalid=REPAIR_INVALID_POLYGONS,
                )
                lost_area, relative_loss, _ = relative_area_loss(baseline_polygon, comp_polygon)
                rows.append(
                    {
                        "baseline_mu_max": float(BASELINE_MU_MAX),
                        "baseline_mu_incr": float(BASELINE_MU_INCR),
                        "baseline_t_dev": int(BASELINE_T_DEV),
                        "comp_mu_max": float(mu_max),
                        "comp_mu_incr": float(mu_incr),
                        "comp_t_dev": int(t_dev),
                        "target_year": int(TARGET_YEAR),
                        "delay_year": int(t_dev_to_year(t_dev)),
                        "q_min": float(bounds.q_min),
                        "q_max": float(bounds.q_max),
                        "temp_min": float(bounds.temp_min),
                        "temp_max": float(bounds.temp_max),
                        "base_area_scaled": float(baseline_area),
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
        dataframe.to_excel(writer, sheet_name="mu_max_mu_incr_loss", index=False)
    return out_path


def _loss_matrix(dataframe: pd.DataFrame, t_dev: int) -> np.ndarray:
    mu_incr_grid = mu_incr_values()
    mu_max_grid = mu_max_values()
    matrix = np.full((len(mu_incr_grid), len(mu_max_grid)), np.nan)
    subset = dataframe[dataframe["comp_t_dev"] == t_dev]
    for i, mu_incr in enumerate(mu_incr_grid):
        for j, mu_max in enumerate(mu_max_grid):
            match = subset[
                np.isclose(subset["comp_mu_incr"], mu_incr)
                & np.isclose(subset["comp_mu_max"], mu_max)
            ]
            if not match.empty:
                matrix[i, j] = float(match.iloc[0]["relative_loss"])
    return matrix


def plot_relative_loss_heatmaps(dataframe: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=150, constrained_layout=True)
    axes = axes.ravel()
    vmax = float(dataframe["relative_loss"].max())
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    image = None
    mu_max_grid = mu_max_values()
    mu_incr_grid = mu_incr_values()

    for idx, t_dev in enumerate(TARGET_T_DEVS):
        ax = axes[idx]
        matrix = _loss_matrix(dataframe, t_dev)
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", norm=norm)
        ax.set_title(f"Delay to {t_dev_to_year(t_dev)}")
        ax.set_xlabel("mu_max")
        ax.set_ylabel("mu_incr")
        ax.set_xticks(range(len(mu_max_grid)))
        ax.set_xticklabels([f"{value:.2f}" for value in mu_max_grid], rotation=45, ha="right")
        ax.set_yticks(range(len(mu_incr_grid)))
        ax.set_yticklabels([f"{value:.2f}" for value in mu_incr_grid])

    fig.suptitle(
        f"Relative loss heatmaps\nbaseline mu_max=1.20, mu_incr=1.10, year 2020, target year {TARGET_YEAR}",
        fontsize=20,
    )
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.88, label="Relative loss")

    out_path = OUTPUT_DIR / f"compensation_mu_max_and_mu_incr_relative_loss_heatmaps_year{TARGET_YEAR}.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_relative_loss_bar3d(dataframe: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(22, 12), dpi=150)
    vmax = float(dataframe["relative_loss"].max())
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    mu_max_grid = mu_max_values()
    mu_incr_grid = mu_incr_values()

    x_positions = np.arange(len(mu_max_grid), dtype=float)
    y_positions = np.arange(len(mu_incr_grid), dtype=float)
    xpos_grid, ypos_grid = np.meshgrid(x_positions, y_positions)

    for idx, t_dev in enumerate(TARGET_T_DEVS, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        matrix = _loss_matrix(dataframe, t_dev)
        xpos = xpos_grid.ravel()
        ypos = ypos_grid.ravel()
        zpos = np.zeros_like(xpos)
        dx = np.full_like(xpos, 0.65, dtype=float)
        dy = np.full_like(ypos, 0.65, dtype=float)
        dz = np.nan_to_num(matrix, nan=0.0).ravel()
        bar_colors = cmap(norm(dz))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colors, shade=True)
        ax.set_title(f"Delay to {t_dev_to_year(t_dev)}")
        ax.set_xticks(x_positions + 0.325)
        ax.set_xticklabels([f"{value:.2f}" for value in mu_max_grid], rotation=45, ha="right")
        ax.set_yticks(y_positions + 0.325)
        ax.set_yticklabels([f"{value:.2f}" for value in mu_incr_grid])
        ax.set_xlabel("mu_max")
        ax.set_ylabel("mu_incr")
        ax.set_zlabel("Rel. loss")
        ax.view_init(elev=26, azim=-58)

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.94, wspace=0.12, hspace=0.18)
    out_path = OUTPUT_DIR / f"compensation_mu_max_and_mu_incr_relative_loss_bar3d_year{TARGET_YEAR}.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_relative_loss_line_slices(dataframe: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=150, constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("plasma")
    mu_max_grid = mu_max_values()
    mu_incr_grid = mu_incr_values()

    for idx, t_dev in enumerate(TARGET_T_DEVS):
        ax = axes[idx]
        matrix = _loss_matrix(dataframe, t_dev)
        for row_idx, mu_incr in enumerate(mu_incr_grid):
            ax.plot(
                mu_max_grid,
                matrix[row_idx, :],
                marker="o",
                color=cmap(row_idx / max(1, len(mu_incr_grid) - 1)),
                label=f"mu_incr={mu_incr:.2f}",
            )
        ax.set_title(f"Delay to {t_dev_to_year(t_dev)}")
        ax.set_xlabel("mu_max")
        ax.set_ylabel("Relative loss")
        ax.set_ylim(bottom=0.0)
        ax.grid(True)

    axes[0].legend(loc="best", ncols=2)
    fig.suptitle(
        f"Relative-loss slices by mu_incr\nbaseline mu_max=1.20, mu_incr=1.10, year 2020, target year {TARGET_YEAR}",
        fontsize=20,
    )

    out_path = OUTPUT_DIR / f"compensation_mu_max_and_mu_incr_relative_loss_lines_year{TARGET_YEAR}.png"
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    rows = compute_loss_rows()
    dataframe = pd.DataFrame(rows)

    xlsx_path = save_loss_table_xlsx(
        rows,
        OUTPUT_DIR / f"compensation_mu_max_and_mu_incr_loss_table_year{TARGET_YEAR}.xlsx",
    )
    heatmap_path = plot_relative_loss_heatmaps(dataframe)
    bar3d_path = plot_relative_loss_bar3d(dataframe)
    line_path = plot_relative_loss_line_slices(dataframe)

    print(f"Saved loss table: {xlsx_path}")
    print(f"Saved heatmap panel: {heatmap_path}")
    print(f"Saved 3D-bar panel: {bar3d_path}")
    print(f"Saved line-slice panel: {line_path}")
    print(
        "Relative-loss range: "
        f"{float(dataframe['relative_loss'].min()):.6f} .. "
        f"{float(dataframe['relative_loss'].max()):.6f}"
    )


if __name__ == "__main__":
    main()
