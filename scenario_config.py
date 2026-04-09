import numpy as np
import os


SCENARIO_CHANGE_MU_MAX = "change_mu_max"
SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT = "change_mu_max_no_constraint"
SCENARIO_CHANGE_MU_MAX_AND_MU_INCR = "change_mu_max_and_mu_incr"
SCENARIO_CHANGE_T_TAR = "change_t_tar"
SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU = "change_mu_max_optimal_mu"
SCENARIO_CHANGE_MU_INCR = "change_mu_incr"


DEFAULT_T_TAR = 17
DEFAULT_MU_MAX = 1.2
DEFAULT_MU_INCR = 1.1
MU_INCR_NO_CONSTRAINT = "no_constraint"


MU_MAX_RANGE = np.linspace(1.2, 2.1, 10)
MU_MAX_AND_MU_INCR_MU_MAX_RANGE = np.linspace(1.2, 2.0, 9)
T_TAR_RANGE = np.arange(17, 20, 1)
MU_MAX_OPTIMAL_RANGE = np.arange(1.2, 0.99, -0.05)
MU_INCR_RANGE = [1.1, 1.2, 1.3, 1.4, 1.5, MU_INCR_NO_CONSTRAINT]
MU_MAX_AND_MU_INCR_MU_INCR_RANGE = np.linspace(1.1, 1.7, 7)
MU_MAX_NO_CONSTRAINT_ENV = "NCC_MU_MAX_NO_CONSTRAINT_VALUES"
MU_INCR_ENV = "NCC_MU_INCR_VALUES"
MU_MAX_AND_MU_INCR_MU_MAX_ENV = "NCC_MU_MAX_AND_MU_INCR_MU_MAX_VALUES"
MU_MAX_AND_MU_INCR_MU_INCR_ENV = "NCC_MU_MAX_AND_MU_INCR_MU_INCR_VALUES"


def _is_close_or_equal(value, baseline):
    if isinstance(value, (tuple, list, np.ndarray)) or isinstance(baseline, (tuple, list, np.ndarray)):
        if not (isinstance(value, (tuple, list, np.ndarray)) and isinstance(baseline, (tuple, list, np.ndarray))):
            return False
        if len(value) != len(baseline):
            return False
        return all(_is_close_or_equal(v_item, b_item) for v_item, b_item in zip(value, baseline))
    if isinstance(value, str) or isinstance(baseline, str):
        return value == baseline
    return np.isclose(value, baseline)


def t_dev_runs(value, baseline):
    if _is_close_or_equal(value, baseline):
        return [1, 4,5,6,7,8,9 ] # [1, 4, 5, 6, 7]
    return [4,5,6,7,8,9]


def data_filename(scenario_name, sweep_value, t_dev):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}_muincr_none.csv"
    if scenario_name == SCENARIO_CHANGE_MU_MAX_AND_MU_INCR:
        mu_max, mu_incr = sweep_value
        return f"plots_data/NCC_mu{mu_max:.2f}_muincr{mu_incr:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return f"plots_data/NCC_t_tar{sweep_value:.2f}_tdev{t_dev}.csv"
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return f"plots_data/NCC_mu{sweep_value:.2f}_tdev{t_dev}_mu_opt.csv"
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        if sweep_value == MU_INCR_NO_CONSTRAINT:
            return f"plots_data/NCC_muincr_none_tdev{t_dev}.csv"
        return f"plots_data/NCC_muincr{sweep_value:.2f}_tdev{t_dev}.csv"
    raise ValueError(f"Unknown scenario: {scenario_name}")


def _parse_float_list_from_env(env_name):
    raw = os.environ.get(env_name)
    if not raw:
        return None

    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))

    if not values:
        raise ValueError(f"{env_name} was provided but no numeric values were found.")

    return np.array(values, dtype=float)


def _parse_mu_incr_list_from_env(env_name):
    raw = os.environ.get(env_name)
    if not raw:
        return None

    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == MU_INCR_NO_CONSTRAINT:
            values.append(MU_INCR_NO_CONSTRAINT)
        else:
            values.append(float(chunk))

    if not values:
        raise ValueError(f"{env_name} was provided but no mu_incr values were found.")

    return values


def scenario_sweep_values(scenario_name):
    if scenario_name == SCENARIO_CHANGE_MU_MAX:
        return MU_MAX_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT:
        override = _parse_float_list_from_env(MU_MAX_NO_CONSTRAINT_ENV)
        return override if override is not None else MU_MAX_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_MAX_AND_MU_INCR:
        mu_max_values = _parse_float_list_from_env(MU_MAX_AND_MU_INCR_MU_MAX_ENV)
        mu_incr_values = _parse_float_list_from_env(MU_MAX_AND_MU_INCR_MU_INCR_ENV)
        mu_max_values = mu_max_values if mu_max_values is not None else MU_MAX_AND_MU_INCR_MU_MAX_RANGE
        mu_incr_values = mu_incr_values if mu_incr_values is not None else MU_MAX_AND_MU_INCR_MU_INCR_RANGE
        return [
            (float(mu_max), float(mu_incr))
            for mu_incr in mu_incr_values
            for mu_max in mu_max_values
        ]
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return T_TAR_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU:
        return MU_MAX_OPTIMAL_RANGE
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        override = _parse_mu_incr_list_from_env(MU_INCR_ENV)
        return override if override is not None else MU_INCR_RANGE
    raise ValueError(f"Unknown scenario: {scenario_name}")


def scenario_baseline(scenario_name):
    if scenario_name == SCENARIO_CHANGE_T_TAR:
        return DEFAULT_T_TAR
    if scenario_name == SCENARIO_CHANGE_MU_INCR:
        return DEFAULT_MU_INCR
    if scenario_name == SCENARIO_CHANGE_MU_MAX_AND_MU_INCR:
        return (DEFAULT_MU_MAX, DEFAULT_MU_INCR)
    return DEFAULT_MU_MAX


def scenario_t_dev_runs(scenario_name, sweep_value):
    return t_dev_runs(sweep_value, scenario_baseline(scenario_name))
