import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# === Define plot styles for each t_dev ===
# list_of_runs = [13, 9, 5, 1]
list_of_runs = [9, 7, 5, 1]
# lince_col_dic = {1:'y', 5:'r', 9:'b', 13:'g'}
lince_col_dic = {1:'y', 5:'r', 7:'b', 9:'g'}
# marker_dic = {1:'o', 5:'<', 9:'>', 13:'o'}
marker_dic = {1:'o', 5:'<', 7:'>', 9:'o'}
plt.rcParams.update({
    'font.size': 14,           # Bigger font size
    'axes.labelsize': 16,      # Axis label size
    'axes.titlesize': 18,      # Title font size
    'legend.fontsize': 14,     # Legend font size
    'lines.linewidth': 2.5,    # Thicker lines
    'lines.markersize': 6      # Bigger markers
})

if __name__ == '__main__':
    for t_dev in list_of_runs:
        # Load convex hull points from CSV
        csv_path = f'plots_data/NCC_DICE_plots_data/{t_dev}.csv'
        try:
            points = np.loadtxt(csv_path, delimiter=",")
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
            continue

        # Scatter points
        plt.plot(points[:, 0], points[:, 1],
                 linestyle='None',
                 marker=marker_dic[t_dev],
                 markersize=2,
                 color=lince_col_dic[t_dev],
                 label=str(2015 + 5 * t_dev))

        # Draw convex hull line
        polygon = Polygon(points, fill=False, edgecolor=lince_col_dic[t_dev], linewidth=1.5)
        plt.gca().add_patch(polygon)

    # General plot settings
    plt.xlabel("Y")
    plt.ylabel("T")
    plt.title("Negative emissions 1.5 speed inf (new years)")  # or change to match your miu_incr mode
    plt.ylim([0, 4])
    plt.xlim([570, 900])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.yticks(np.linspace(0, 4, 5))   # Adjust limits and steps as needed
    plt.xticks(np.linspace(570, 900, 6))  # Adjust as needed
    plt.tight_layout()
    plt.show()


# Plotting mu_opt case
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pathlib import Path

# === Plot style settings ===
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
    'lines.markersize': 6
})

# === Fixed t_dev = 1, mu_max = 1.2 to include in all plots ===
REFERENCE_T_DEV = 1
REFERENCE_MU_MAX = 1.2

# === List of t_dev values ===
all_t_devs = range(2, 8)  # for mu_max > 1.2
reference_label = f"Year {2015 + 5 * REFERENCE_T_DEV}, μ_max={REFERENCE_MU_MAX:.2f}"

# === Color and marker styles by t_dev ===
color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive']
marker_list = ['o', '<', '>', '^', 's', 'D', 'P', '*', 'x']
style_map = {
    t: (color_list[(t - 1) % len(color_list)], marker_list[(t - 1) % len(marker_list)])
    for t in range(1, 10)
}

# === Load reference points once ===
ref_filename = f"plots_data/NCC_mu{REFERENCE_MU_MAX:.2f}_tdev{REFERENCE_T_DEV}_mu_opt.csv"
try:
    ref_points = np.loadtxt(ref_filename, delimiter=",")
except Exception as e:
    raise RuntimeError(f"Cannot load reference file: {ref_filename}\n{e}")

# === Create separate plot for each mu_max ===
# mu_max_values = np.arange(1.2, 1.51, 0.05)
# mu_max_values = np.arange(1.2, 9.21, 1)
mu_max_values = np.arange(1.2, 0.99, -0.05)
output_dir = Path("plots_png")
output_dir.mkdir(exist_ok=True)

for mu_max in mu_max_values:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # 1. Plot reference t_dev = 1 for mu_max = 1.2
    ref_color, ref_marker = style_map[REFERENCE_T_DEV]
    ax.plot(ref_points[:, 0], ref_points[:, 1],
            linestyle='None',
            marker=ref_marker,
            color=ref_color,
            label=reference_label)
    ref_poly = Polygon(ref_points, fill=False, edgecolor=ref_color, linewidth=1.2)
    ax.add_patch(ref_poly)

    # 2. Plot all other t_devs for current mu_max
    for t_dev in all_t_devs:
        filename = f"plots_data/NCC_mu{mu_max:.2f}_tdev{t_dev}_mu_opt.csv"
        try:
            points = np.loadtxt(filename, delimiter=",")
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            continue

        color, marker = style_map[t_dev]
        ax.plot(points[:, 0], points[:, 1],
                linestyle='None',
                # marker=marker,
                color=color,
                label=f"Year {2015 + 5 * t_dev}, μ_max={mu_max:.2f}")
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=1.2)
        ax.add_patch(polygon)

    # === Final plot settings ===
    ax.set_xlabel("Q in 2100")
    ax.set_ylabel("ΔT in 2100 (°C)")
    ax.set_title(f"RS for μ_max = {mu_max:.2f}")
    ax.set_ylim([0, 4])
    ax.set_xlim([570, 900])
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 4, 5))
    ax.set_xticks(np.linspace(570, 900, 6))
    plt.tight_layout()

    # === Save figure ===
    plot_path = output_dir / f"RS_mu{mu_max:.2f}.png"
    plt.savefig(plot_path)
    plt.close()

print("All plots saved to:", output_dir.resolve())


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Polygon
# from pathlib import Path

# # === Plot style settings ===
# plt.rcParams.update({
#     'font.size': 14,
#     'axes.labelsize': 16,
#     'axes.titlesize': 18,
#     'legend.fontsize': 12,
#     'lines.linewidth': 2.0,
#     'lines.markersize': 6
# })

# # === Fixed t_dev = 1, mu_max = 1.2 to include in all plots ===
# REFERENCE_T_DEV = 1
# REFERENCE_T_TAR = 17

# # === List of t_dev values ===
# all_t_devs = range(2, 8)  # for mu_max > 1.2
# reference_label = f"Year {2015 + 5 * REFERENCE_T_DEV}, t_tar={2015 + 5 * REFERENCE_T_TAR:.2f}"

# # === Color and marker styles by t_dev ===
# color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive']
# marker_list = ['o', '<', '>', '^', 's', 'D', 'P', '*', 'x']
# style_map = {
#     t: (color_list[(t - 1) % len(color_list)], marker_list[(t - 1) % len(marker_list)])
#     for t in range(1, 10)
# }

# # === Load reference points once ===
# ref_filename = f"plots_data/NCC_t_tar{REFERENCE_T_TAR:.2f}_tdev{REFERENCE_T_DEV}.csv"
# try:
#     ref_points = np.loadtxt(ref_filename, delimiter=",")
# except Exception as e:
#     raise RuntimeError(f"Cannot load reference file: {ref_filename}\n{e}")

# # === Create separate plot for each t_tar ===
# # mu_max_values = np.arange(1.2, 1.51, 0.05)
# t_tar_values = np.arange(17, 20, 1)
# output_dir = Path("plots_png")
# output_dir.mkdir(exist_ok=True)

# for t_tar in t_tar_values:
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

#     # 1. Plot reference t_dev = 1 for t_tar = 17
#     ref_color, ref_marker = style_map[REFERENCE_T_DEV]
#     ax.plot(ref_points[:, 0], ref_points[:, 1],
#             linestyle='None',
#             marker=ref_marker,
#             color=ref_color,
#             label=reference_label)
#     ref_poly = Polygon(ref_points, fill=False, edgecolor=ref_color, linewidth=1.2)
#     ax.add_patch(ref_poly)

#     # 2. Plot all other t_devs for current t_tar
#     for t_dev in all_t_devs:
#         filename = f"plots_data/NCC_t_tar{t_tar:.2f}_tdev{t_dev}.csv"
#         try:
#             points = np.loadtxt(filename, delimiter=",")
#         except Exception as e:
#             print(f"Could not load {filename}: {e}")
#             continue

#         color, marker = style_map[t_dev]
#         ax.plot(points[:, 0], points[:, 1],
#                 linestyle='None',
#                 marker=marker,
#                 color=color,
#                 label=f"Year {2015 + 5 * t_dev}, T={2015+5*t_tar}")
#         polygon = Polygon(points, fill=False, edgecolor=color, linewidth=1.2)
#         ax.add_patch(polygon)

#     # === Final plot settings ===
#     ax.set_xlabel("Q in 2100")
#     ax.set_ylabel("ΔT in 2100 (°C)")
#     ax.set_title(f"RS for T = {2015+5*t_tar:.2f}")
#     ax.set_ylim([0, 4])
#     ax.set_xlim([570, 1020])
#     ax.legend(loc='upper right', fontsize=10, frameon=True)
#     ax.grid(True)
#     ax.set_yticks(np.linspace(0, 4, 5))
#     ax.set_xticks(np.linspace(570, 1020, 6))
#     plt.tight_layout()

#     # === Save figure ===
#     plot_path = output_dir / f"RS_t_tar{t_tar:.2f}.png"
#     plt.savefig(plot_path)
#     plt.close()

# print("All plots saved to:", output_dir.resolve())