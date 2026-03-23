import numpy as np


SCENARIO_CHANGE_MU_MAX = "change_mu_max"
SCENARIO_CHANGE_T_TAR = "change_t_tar"
SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU = "change_mu_max_optimal_mu"
SCENARIO_CHANGE_MU_INCR = "change_mu_incr"


DEFAULT_T_TAR = 17
DEFAULT_MU_MAX = 1.2
DEFAULT_MU_INCR = 1.1
MU_INCR_NO_CONSTRAINT = "no_constraint"


MU_MAX_RANGE = np.linspace(1.2, 1.4, 1)
T_TAR_RANGE = np.arange(17, 20, 1)
MU_MAX_OPTIMAL_RANGE = np.arange(1.2, 0.99, -0.05)
MU_INCR_RANGE = [1.1, 1.2, 1.3, 1.4, 1.5, MU_INCR_NO_CONSTRAINT]


def _is_close_or_equal(value, baseline):
    if isinstance(value, str) or isinstance(baseline, str):
        return value == baseline
    return np.isclose(value, baseline)


def t_dev_runs(value, baseline):
    if _is_close_or_equal(value, baseline):
        return [1, 5, 6, 7] # [1, 4, 5, 6, 7]
    return [5, 6, 7]


def data_filename(scenario_name, sweep_value, t_dev):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return f"plots_data/NCC_t_tar{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}_mu_opt.csv"
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        if sweep_value == MU_INCR_NO_CONSTRAINT:
            return f"plots_data/NCC_muincr_none_tdev{t_dev}.csv"
        return f"plots_data/NCC_muincr{sweep_value:.2f}_tdev{t_dev}.csv"
    raise ValueError(f"Unknown scenario: {scenario_name}")


def scenario_sweep_values(scenario_name):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return MU_MAX_RANGE
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return T_TAR_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return MU_MAX_OPTIMAL_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        return MU_INCR_RANGE
    raise ValueError(f"Unknown scenario: {scenario_name}")


def scenario_baseline(scenario_name):
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return DEFAULT_T_TAR
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        return DEFAULT_MU_INCR
    return DEFAULT_MU_MAX
