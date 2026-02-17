import numpy as np


SCENARIO_CHANGE_MU_MAX = "change_mu_max"
SCENARIO_CHANGE_T_TAR = "change_t_tar"
SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU = "change_mu_max_optimal_mu"


DEFAULT_T_TAR = 17
DEFAULT_MU_MAX = 1.2


MU_MAX_RANGE = np.linspace(1.2, 2.21, 5)
T_TAR_RANGE = np.arange(17, 20, 1)
MU_MAX_OPTIMAL_RANGE = np.arange(1.2, 0.99, -0.05)


def t_dev_runs(value, baseline):
    if np.isclose(value, baseline):
        return [1, 2, 3, 4, 5, 6, 7]
    return [2, 3, 4, 5, 6, 7]


def data_filename(scenario_name, sweep_value, t_dev):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return f"plots_data/NCC_t_tar{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}_mu_opt.csv"
    raise ValueError(f"Unknown scenario: {scenario_name}")


def scenario_sweep_values(scenario_name):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return MU_MAX_RANGE
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return T_TAR_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return MU_MAX_OPTIMAL_RANGE
    raise ValueError(f"Unknown scenario: {scenario_name}")


def scenario_baseline(scenario_name):
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return DEFAULT_T_TAR
    return DEFAULT_MU_MAX
