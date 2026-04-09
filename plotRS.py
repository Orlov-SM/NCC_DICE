"""
# the code consist of
# 1) function DICE16_on_path that computes trajectories
# 2) function run_boundary that computes boundary using DICE16_on_path
# 3) using 2) for normalization,
# function RS_bulk computes and saves the hull of RS; also plots figures

# conda activate /anaconda2/envs/RFBR; ipython
# import os
# os.chdir(r'/Users/ab/OneDrive - IIASA/TG_model/DICE_ERM/D16_RS/')
# execfile('on_path_DICEbulkRS.py')

with open("plotRS.py") as f:
    code = compile(f.read(), "plotRS.py", 'exec')
    exec(code)
"""

import os
import sys

# Keep Matplotlib cache writable when running from environments where the home
# config dir is unavailable, which otherwise slows or blocks multiprocessing.
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.mplconfig'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# On macOS, the parallel IPOPT runs can oversubscribe CPU threads through
# vecLib/BLAS unless each worker is forced to stay single-threaded.
if sys.platform == 'darwin':
    for env_name in (
        'VECLIB_MAXIMUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'BLIS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(env_name, '1')

import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from importlib import reload
import datetime
import pickle
import scipy



import RSroutines
reload(RSroutines)
import scenario_config as sc



####################################
## i) This part computes RS at t_tar for the ***NCC paper*** for
## all deviations from the BAU path at specified time instant t_dev
####################################
remove_mu_constr = False # if True, disable post-2160 incremental mu constraint
mu_incr = sc.DEFAULT_MU_INCR
dam_fun_Weitzman = False #select the damage function




# resol1 =  (1, 150) #defines the level of approximation, higher product of this numbers gives better approximation
resol1 =  (1, 30) # (1, 80) - for nice plots


t_start = datetime.datetime.now()
# miu_incr = 0.15 #constraint for incremental change of MIU
# Scenario switches: enable exactly one.
RUN_CHANGE_MU_MAX = True
RUN_CHANGE_MU_MAX_NO_CONSTRAINT = False
RUN_CHANGE_MU_MAX_AND_MU_INCR = False
RUN_CHANGE_T_TAR = False
RUN_CHANGE_MU_MAX_OPTIMAL_MU = False
RUN_CHANGE_MU_INCR = False
RS_MODE = 'parallel_grid'  # 'grid', 'parallel_grid', 'parallel', 'mod_util', 'ray'


_SCENARIO_ENV_NAME = "NCC_SCENARIO"
_SCENARIO_FLAG_BY_NAME = {
    sc.SCENARIO_CHANGE_MU_MAX: "RUN_CHANGE_MU_MAX",
    sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT: "RUN_CHANGE_MU_MAX_NO_CONSTRAINT",
    sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR: "RUN_CHANGE_MU_MAX_AND_MU_INCR",
    sc.SCENARIO_CHANGE_T_TAR: "RUN_CHANGE_T_TAR",
    sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU: "RUN_CHANGE_MU_MAX_OPTIMAL_MU",
    sc.SCENARIO_CHANGE_MU_INCR: "RUN_CHANGE_MU_INCR",
}


def _apply_scenario_override_from_env():
    override = os.environ.get(_SCENARIO_ENV_NAME)
    if not override:
        return
    if override not in _SCENARIO_FLAG_BY_NAME:
        allowed = ", ".join(sorted(_SCENARIO_FLAG_BY_NAME))
        raise ValueError(
            f"Unsupported {_SCENARIO_ENV_NAME}={override!r}. Allowed values: {allowed}"
        )

    for flag_name in _SCENARIO_FLAG_BY_NAME.values():
        globals()[flag_name] = False
    globals()[_SCENARIO_FLAG_BY_NAME[override]] = True


_apply_scenario_override_from_env()


def _run_single_case(t_dev, t_tar, mu_max, out_name, opt=False, mu_opt=None, remove_mu_constr=False, mu_incr=1.1):
    X_t, Y_t = RSroutines.RS_bulk_ncc(
        None, t_dev, t_tar,
        remove_mu_constr=remove_mu_constr,
        dam_fun_Weitzman=dam_fun_Weitzman,
        resol=resol1, mode=RS_MODE,
        mu_max=mu_max,
        opt=opt,
        mu_opt=mu_opt if mu_opt is not None else np.arange(20),
        mu_incr=mu_incr,
    )

    points = np.column_stack((X_t, Y_t))
    hull = scipy.spatial.ConvexHull(points)
    np.savetxt(out_name, points[hull.vertices, :], delimiter=',')


if __name__ == '__main__':
    scenario_flags = [
        RUN_CHANGE_MU_MAX,
        RUN_CHANGE_MU_MAX_NO_CONSTRAINT,
        RUN_CHANGE_MU_MAX_AND_MU_INCR,
        RUN_CHANGE_T_TAR,
        RUN_CHANGE_MU_MAX_OPTIMAL_MU,
        RUN_CHANGE_MU_INCR,
    ]
    if sum(scenario_flags) != 1:
        raise ValueError(
            "Enable exactly one scenario flag: "
            "RUN_CHANGE_MU_MAX, RUN_CHANGE_MU_MAX_NO_CONSTRAINT, RUN_CHANGE_MU_MAX_AND_MU_INCR, "
            "RUN_CHANGE_T_TAR, RUN_CHANGE_MU_MAX_OPTIMAL_MU, RUN_CHANGE_MU_INCR."
        )

    if RUN_CHANGE_MU_MAX:
        mu_max_range = sc.scenario_sweep_values(sc.SCENARIO_CHANGE_MU_MAX)
        t_tar = sc.DEFAULT_T_TAR
        for mu_max in mu_max_range:
            for t_dev in sc.scenario_t_dev_runs(sc.SCENARIO_CHANGE_MU_MAX, mu_max):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX, mu_max, t_dev)
                _run_single_case(t_dev, t_tar, mu_max, out_name, remove_mu_constr=remove_mu_constr, mu_incr=mu_incr)

    elif RUN_CHANGE_MU_MAX_NO_CONSTRAINT:
        mu_max_range = sc.scenario_sweep_values(sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT)
        t_tar = sc.DEFAULT_T_TAR
        for mu_max in mu_max_range:
            for t_dev in sc.scenario_t_dev_runs(sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT, mu_max):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX_NO_CONSTRAINT, mu_max, t_dev)
                _run_single_case(
                    t_dev,
                    t_tar,
                    mu_max,
                    out_name,
                    remove_mu_constr=True,
                    mu_incr=sc.DEFAULT_MU_INCR,
                )

    elif RUN_CHANGE_MU_MAX_AND_MU_INCR:
        sweep_values = sc.scenario_sweep_values(sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR)
        t_tar = sc.DEFAULT_T_TAR
        for mu_max, mu_incr_value in sweep_values:
            for t_dev in sc.scenario_t_dev_runs(
                sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR,
                (mu_max, mu_incr_value),
            ):
                out_name = sc.data_filename(
                    sc.SCENARIO_CHANGE_MU_MAX_AND_MU_INCR,
                    (mu_max, mu_incr_value),
                    t_dev,
                )
                _run_single_case(
                    t_dev,
                    t_tar,
                    mu_max,
                    out_name,
                    remove_mu_constr=False,
                    mu_incr=mu_incr_value,
                )

    elif RUN_CHANGE_T_TAR:
        mu_max = sc.DEFAULT_MU_MAX
        t_tar_range = sc.T_TAR_RANGE
        for t_tar in t_tar_range:
            for t_dev in sc.scenario_t_dev_runs(sc.SCENARIO_CHANGE_T_TAR, t_tar):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_T_TAR, t_tar, t_dev)
                _run_single_case(t_dev, t_tar, mu_max, out_name, remove_mu_constr=remove_mu_constr, mu_incr=mu_incr)

    elif RUN_CHANGE_MU_MAX_OPTIMAL_MU:
        t_tar = sc.DEFAULT_T_TAR
        _, mu_opt, _ = RSroutines.solve_vanilla_optimal_path(duration=t_tar + 1)
        if mu_opt is None:
            raise RuntimeError("Failed to compute optimal path for mu_opt.")

        mu_max_range = sc.MU_MAX_OPTIMAL_RANGE
        for mu_max in mu_max_range:
            for t_dev in sc.scenario_t_dev_runs(sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU, mu_max):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU, mu_max, t_dev)
                _run_single_case(
                    t_dev,
                    t_tar,
                    mu_max,
                    out_name,
                    opt=True,
                    mu_opt=mu_opt,
                    remove_mu_constr=remove_mu_constr,
                    mu_incr=mu_incr,
                )

    elif RUN_CHANGE_MU_INCR:
        t_tar = sc.DEFAULT_T_TAR
        mu_max = sc.DEFAULT_MU_MAX
        for mu_incr_value in sc.MU_INCR_RANGE:
            rm_constraint = mu_incr_value == sc.MU_INCR_NO_CONSTRAINT
            mu_incr_numeric = sc.DEFAULT_MU_INCR if rm_constraint else float(mu_incr_value)
            for t_dev in sc.scenario_t_dev_runs(sc.SCENARIO_CHANGE_MU_INCR, mu_incr_value):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_INCR, mu_incr_value, t_dev)
                _run_single_case(
                    t_dev,
                    t_tar,
                    mu_max,
                    out_name,
                    remove_mu_constr=rm_constraint,
                    mu_incr=mu_incr_numeric,
                )

    print("Total time:", datetime.datetime.now() - t_start)
