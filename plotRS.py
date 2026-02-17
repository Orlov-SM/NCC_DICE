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

import matplotlib.pyplot as plt
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
miu_incr = False #no incremental constraints
dam_fun_Weitzman = False #select the damage function




# resol1 =  (1, 150) #defines the level of approximation, higher product of this numbers gives better approximation
resol1 =  (1, 10)


t_start = datetime.datetime.now()
# miu_incr = 0.15 #constraint for incremental change of MIU
extreme_res = {}

# Scenario switches: enable exactly one.
RUN_CHANGE_MU_MAX = True
RUN_CHANGE_T_TAR = False
RUN_CHANGE_MU_MAX_OPTIMAL_MU = False
RS_MODE = 'grid'  # 'grid', 'parallel_grid', 'parallel', 'mod_util', 'ray'


def _run_single_case(t_dev, t_tar, mu_max, out_name, opt=False, mu_opt=None):
    extreme_res[t_dev] = (1, 3.5, 580, 850)
    X_t, Y_t = RSroutines.RS_bulk_ncc(
        extreme_res[t_dev], t_dev, t_tar,
        remove_mu_constr=miu_incr,
        dam_fun_Weitzman=dam_fun_Weitzman,
        resol=resol1, mode=RS_MODE,
        mu_max=mu_max,
        opt=opt,
        mu_opt=mu_opt if mu_opt is not None else np.arange(20),
    )

    points = np.column_stack((X_t, Y_t))
    hull = scipy.spatial.ConvexHull(points)
    np.savetxt(out_name, points[hull.vertices, :], delimiter=',')


if __name__ == '__main__':
    scenario_flags = [
        RUN_CHANGE_MU_MAX,
        RUN_CHANGE_T_TAR,
        RUN_CHANGE_MU_MAX_OPTIMAL_MU,
    ]
    if sum(scenario_flags) != 1:
        raise ValueError(
            "Enable exactly one scenario flag: "
            "RUN_CHANGE_MU_MAX, RUN_CHANGE_T_TAR, RUN_CHANGE_MU_MAX_OPTIMAL_MU."
        )

    if RUN_CHANGE_MU_MAX:
        mu_max_range = sc.MU_MAX_RANGE
        t_tar = sc.DEFAULT_T_TAR
        for mu_max in mu_max_range:
            for t_dev in sc.t_dev_runs(mu_max, baseline=sc.DEFAULT_MU_MAX):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX, mu_max, t_dev)
                _run_single_case(t_dev, t_tar, mu_max, out_name)

    elif RUN_CHANGE_T_TAR:
        mu_max = sc.DEFAULT_MU_MAX
        t_tar_range = sc.T_TAR_RANGE
        for t_tar in t_tar_range:
            for t_dev in sc.t_dev_runs(t_tar, baseline=sc.DEFAULT_T_TAR):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_T_TAR, t_tar, t_dev)
                _run_single_case(t_dev, t_tar, mu_max, out_name)

    elif RUN_CHANGE_MU_MAX_OPTIMAL_MU:
        t_tar = sc.DEFAULT_T_TAR
        _, mu_opt, _ = RSroutines.solve_vanilla_optimal_path(duration=t_tar + 1)
        if mu_opt is None:
            raise RuntimeError("Failed to compute optimal path for mu_opt.")

        mu_max_range = sc.MU_MAX_OPTIMAL_RANGE
        for mu_max in mu_max_range:
            for t_dev in sc.t_dev_runs(mu_max, baseline=sc.DEFAULT_MU_MAX):
                out_name = sc.data_filename(sc.SCENARIO_CHANGE_MU_MAX_OPTIMAL_MU, mu_max, t_dev)
                _run_single_case(t_dev, t_tar, mu_max, out_name, opt=True, mu_opt=mu_opt)

    print("Total time:", datetime.datetime.now() - t_start)
