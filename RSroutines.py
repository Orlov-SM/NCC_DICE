## In this code we introduce the function that
## i) computes RS at t_tar of all deviations from the BAU path at specified time instant t_dev
## ii)computes RS at t_tar of all deviations from the BAU path at specified time instant t_dev WITH different external debt q_debt
## iii)computes RS at t_tar of all deviations from the BAU path at specified time instant t_dev WITH HIGHER limit for mitigation q_miu
## iv???

# #p3.7
# with open("RSroutines.py") as f:
#     code = compile(f.read(), "RSroutines.py", 'exec')
#     exec(code)


import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *
from pyomo.dae import *
from importlib import reload
import ncc_rs
reload(ncc_rs)
from concurrent.futures import ProcessPoolExecutor
from pyomo.common.errors import ApplicationError

# here we run DICE subject to the specific contraints as the fraction of utility normolized to [0,1] with coef_list
# that focuses on max/min T/Y
def solve_vanilla_optimal_path(duration=100, disc_points=None, mu_max=1.2):
    if disc_points is None:
        disc_points = list(range(duration + 1))

    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, duration), initialize=disc_points)

    # Required suffixes
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    # Make the vanilla NCC model
    m = ncc_rs.makeNCCModel(m, model_mode='vanilla')
    m.mu_max = mu_max

    # Solve
    solver = SolverFactory("ipopt")
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['acceptable_tol'] = 1e-6
    solver.options['constr_viol_tol'] = 1e-9
    solver.options['max_iter'] = 7000
    results = solver.solve(m, tee=False)

    if (results.solver.status != SolverStatus.ok or
        results.solver.termination_condition != TerminationCondition.optimal):
        print("⚠️ Solver failed or did not reach optimality.")
        print(
            f"Solver status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )
        return None, None, None

    # Extract values
    mu_vals = []
    S_vals = []
    t_vals = []

    for t in sorted(m.t):
        try:
            mu = value(m.mu[t])
            S = value(m.S[t])

            mu_vals.append(mu)
            S_vals.append(S)
            t_vals.append(t)
        except Exception as e:
            print(f"Error at t={t}: {e}")
            mu_vals.append(np.nan)
            S_vals.append(np.nan)
            t_vals.append(t)

    return np.array(t_vals), np.array(mu_vals), np.array(S_vals)

extreme_coef_list = ((0, -1, 0, ), (0, 1, 0,), (0, 0, -1,), (0, 0, 1,),)
def run_boundary_NCC(
    t_dev,
    mu_max,
    t_tar=17,
    coef_list = extreme_coef_list,
    remove_mu_constr=False,
    mu_incr=1.1,
):
    T_list = []
    Y_list = []
    duration = max(20, t_tar + 1)
    dicsr_points = range(duration)
    for coef_list_i in coef_list:
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, duration), initialize = dicsr_points)
        m = ncc_rs.makeNCCModel(
            m,
            model_mode='RS_mod_util',
            tt=t_tar,
            remove_mu_constr=remove_mu_constr,
            mu_incr=mu_incr,
        )
        m.coef1 = coef_list_i[0]
        m.coef2 = coef_list_i[1]
        m.coef3 = coef_list_i[2]
        m.mu_max = mu_max
        for i in m.t:
            if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                m.mu[i].fix(m.mu0)
        # if run_miu_up_to_one:
        # 	m.miu_up = 1.0
        # return  m.coef1*m.UTILITY +  m.coef2*m.TATM[tt] + m.coef3*m.Y[tt]


        # def _utility_fraction(m):
        # 	return inequality(bau_value + (fraction - 0.0001)*(opt_value - bau_value), m.UTILITY ,bau_value + (fraction + 0.0001)*(opt_value - bau_value))
        # m.utility_fraction = Constraint(m, expr = inequality(bau_value + (fraction - 0.0001)*(opt_value - bau_value), m.UTILITY ,bau_value + (fraction + 0.0001)*(opt_value - bau_value)))

        # m.UTILITY.fix(bau_value + fraction *(opt_value - bau_value))
        solver=SolverFactory('ipopt') # bonmin
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['acceptable_tol'] = 1e-6
        solver.options['constr_viol_tol'] = 1e-10
        solver.options['max_iter'] = 700
        try:
            results = solver.solve(m,tee=False)
        except ApplicationError:
            continue
        except Exception:
            continue

        if ((results.solver.status == pyomo.opt.SolverStatus.ok) and
            (results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal)):
            t_val = value(m.TAT_IPCC[t_tar])
            y_val = value(m.Q[t_tar])
            if t_val is not None and y_val is not None:
                T_list.append(t_val)
                Y_list.append(y_val)

    if not T_list or not Y_list:
        raise RuntimeError(
            f"run_boundary_NCC failed to compute valid boundary for t_dev={t_dev}, mu_max={mu_max}, t_tar={t_tar}."
        )
    # return  T_list, Y_list
    return (min(T_list), max(T_list), min(Y_list), max(Y_list),)


#This function is used to compute a RS for NCC mod of the DICE model (similar to RS_bulk)
#for vanilla case 4.25 >= tat_IPCC in  2100 >= 0.8179; 1450 >= Q in 2100 >= 78
#we use either grid approximation, modified utility or ray
def RS_bulk_ncc(
    bounds,
    t_dev,
    t_tar,
    remove_mu_constr,
    dam_fun_Weitzman,
    resol = (32,10,),
    mode = 'mod_util',
    mu_max = 1.2,
    opt = False,
    mu_opt = np.arange(20),
    mu_incr = 1.1,
): #'grid', 'mod_util', 'ray'
    # #here we define 500 years
    # duration = 99 # 300
    # tttstep = 1 #5.0
    # dicsr_points = range(100) #np.arange(0.0, duration, tttstep)
    #but actually we need only 100 years and less
    duration = t_tar+1 # 300
    tttstep = 1 #5.0
    dicsr_points = range(duration) #np.arange(0.0, duration, tttstep)


    if bounds is None:
        try:
            bounds = run_boundary_NCC(
                t_dev=t_dev,
                mu_max=mu_max,
                t_tar=t_tar,
                remove_mu_constr=remove_mu_constr,
                mu_incr=mu_incr,
            )
        except RuntimeError:
            print(
                f"WARNING: dynamic boundary failed for t_dev={t_dev}, "
                f"mu_max={mu_max}, t_tar={t_tar}; using fallback bounds "
                "(0, 4, 0, 880)."
            )
            bounds = (0, 3.6, 0, 880)
    norm = False #to better estimate boundary. NEED adjustment!


    n_rays =  resol[0]*resol[1] #64 #change this value for gridded runs! total number of runs = n_coef1**2
    n_coef1 = resol[0]
    n_coef23 = resol[1]

    Xboundary = {}
    Yboundary = {}

    #to count runs with errors
    badruns_counts = 0
    iter_counts = 0

    # #turn on gridded approximation
    # grid_run = False #True
    if mode == 'grid':
        model_initialized = False
        if not model_initialized:
            m = ConcreteModel()
            m.t = ContinuousSet(bounds=(0, duration), initialize = dicsr_points)
            ### Declare all suffixes
            # Ipopt bound multipliers (obtained from solution)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            # Ipopt bound multipliers (sent to solver)
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
            # Obtain dual solutions from first solve and send to warm start
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m = ncc_rs.makeNCCModel(
                m,
                model_mode='RS_mod_util',
                tt=t_tar,
                remove_mu_constr=remove_mu_constr,
                dam_fun_Weitzman=dam_fun_Weitzman,
                mu_incr=mu_incr,
            )
            if opt:
                for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(mu_opt[i])
            else:
                for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(m.mu0)
            model_initialized = True
        m.coef1 = 0 #we neglect the actual utility
        m.mu_max = mu_max
        for util_control in [(1, 0), (-1,0), (0, -1), (0, 1)]:
            for delta in range(n_rays+1):
                iter_counts += 1
                m.coef2 = util_control[0]
                m.coef3 = util_control[1]
                if util_control[0] == 0:
                    m.TAT_IPCC[t_tar].fix(bounds[0]+ delta*(bounds[1]-bounds[0])/n_rays)
                    m.Q[t_tar].unfix()
                else:
                    m.Q[t_tar].fix(bounds[2]+ delta*(bounds[3]-bounds[2])/n_rays)
                    m.TAT_IPCC[t_tar].unfix()
                solver=SolverFactory('ipopt') # bonmin
                solver.options['halt_on_ampl_error'] = 'yes'
                solver.options['acceptable_tol'] = 1e-6
                solver.options['constr_viol_tol'] = 1e-9
                solver.options['max_iter'] = 700
                try:
                    results = solver.solve(m,tee=False)
                    if (results.solver.status == pyomo.opt.SolverStatus.ok) and (results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal):
                        print('SOLVED OK t_dev = ' + str(t_dev) + '; iter '+ str(iter_counts) + ' out of total ' + str(4*(n_rays+1)))
                    else:
                        print('=( not solved for ')
                        badruns_counts += 1
                        print('br ' + str(badruns_counts) + '; iter '+ str(iter_counts) + ' out of total ' + str(4*(n_rays+1)))
                        continue
                        # scale_dic[exper] = [-1]
                except Exception as e:
                    continue
                Xboundary[(util_control, delta)] = value(m.Q[t_tar])
                Yboundary[(util_control, delta)] = value(m.TAT_IPCC[t_tar])

    elif mode == 'mod_util':
        i=0
        model_initialized = False
        # for coef1 in range(n_coef1):
            # coef1_val = 1 - (coef1 + 1)/float(n_coef1)
        for coef23 in range(n_coef23):
            # try:
            if not model_initialized:
                m = ConcreteModel()
                m.t = ContinuousSet(bounds=(0, duration), initialize = dicsr_points)
                ### Declare all suffixes
                # Ipopt bound multipliers (obtained from solution)
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                # Ipopt bound multipliers (sent to solver)
                m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
                # Obtain dual solutions from first solve and send to warm start
                m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

                m = ncc_rs.makeNCCModel(
                    m,
                    model_mode='RS_mod_util',
                    tt=t_tar,
                    remove_mu_constr=remove_mu_constr,
                    dam_fun_Weitzman=dam_fun_Weitzman,
                    mu_incr=mu_incr,
                )
                # makeNCCModel(m, model_mode = 'vanilla', remove_mu_constr = False, dam_fun_Weitzman = False)
                for i in m.t:
                    if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                        m.mu[i].fix(m.mu0)
                model_initialized = True

            iter_counts += 1
            # m.coef1*m.UTILITY +  m.coef2*m.TATM[t_tar] + m.coef3*m.Q[t_tar]
            # if norm == True:
            # 	print('norm')
            # 	m.coef1 = coef1_val/100.
            # 	m.coef2 = sqrt(1-  coef1_val**2) * sin(2*np.pi*coef23/float(n_coef23))/bounds[1]
            # 	m.coef3 = sqrt(1-  coef1_val**2) * cos(2*np.pi*coef23/float(n_coef23))/bounds[3]
            # else:
                # m.coef1 = coef1_val
                # m.coef2 = sqrt(1-  coef1_val**2) * sin(2*np.pi*coef23/float(n_coef23))
                # m.coef3 = sqrt(1-  coef1_val**2) * cos(2*np.pi*coef23/float(n_coef23))
            m.coef1 = 0
            m.coef2 = sin(2*np.pi*coef23/float(n_coef23))
            m.coef3 = cos(2*np.pi*coef23/float(n_coef23))/800


            solver=SolverFactory('ipopt') # bonmin
            solver.options['halt_on_ampl_error'] = 'yes'
            if norm == True:
                solver.options['acceptable_tol'] = 1e-5
                solver.options['constr_viol_tol'] = 1e-8
            else:
                solver.options['acceptable_tol'] = 1e-4
                solver.options['constr_viol_tol'] = 1e-8
            solver.options['max_iter'] = 1500

            try:
                results = solver.solve(m,tee=True)

                if (results.solver.status == pyomo.opt.SolverStatus.ok) and (results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal):
                    print('SOLVED OK t_dev = ' + str(t_dev) + '; iter '+ str(iter_counts) + ' out of total ' +str(n_coef1 * n_coef23) + ' ( br ' + str(badruns_counts)+')' )
                else:
                    print('=( not solved for ')
                    badruns_counts += 1
                    print('br ' + str(badruns_counts) + '; iter '+ str(iter_counts) + ' out of total ' +str(n_coef1 * n_coef23))
                    continue
                    # scale_dic[exper] = [-1]
            except Exception:
                continue
            # Xboundary[(coef1, coef23)] = value(m.K[t_tar])
            Xboundary[ coef23] = value(m.Q[t_tar])
            Yboundary[ coef23] = value(m.TAT_IPCC[t_tar])
            # Xboundary[(coef1, coef23)] = value(m.Ychange_av)
            # Yboundary[(coef1, coef23)] = value(m.TATMav)
            # Zboundary[(coef1, coef23)] = value(m.UTILITY)
            i += 1
            if i%20 == 0:
                print(str(i) + ' out of ' + str(n_coef1* n_coef23))
            # # except:
            # 	if BAU_RUN == True:
            # 		test_var_duals(m, thr = 0.001)
            # 		sys.exit()
            #
            # 	# continue
            # print(badruns_counts)
    elif mode == 'parallel_grid':
        util_controls = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        with ProcessPoolExecutor(max_workers=min(4, len(util_controls))) as executor:
            futures = [
                executor.submit(
                    solve_instance_grid_sweep,
                    uc,
                    bounds,
                    t_dev,
                    duration,
                    dicsr_points,
                    n_rays,
                    mu_max,
                    t_tar,
                    remove_mu_constr,
                    dam_fun_Weitzman,
                    mu_incr,
                )
                for uc in util_controls
            ]
            total = len(util_controls) * (n_rays + 1)
            done = 0
            for future in futures:
                uc, sweep_results = future.result()
                for delta, q_val, tat_val, success in sweep_results:
                    done += 1
                    if success:
                        Xboundary[(uc, delta)] = q_val
                        Yboundary[(uc, delta)] = tat_val
                    else:
                        badruns_counts += 1
                        print(f"Failed for util_control={uc}, delta={delta}")
                    if done % 20 == 0:
                        print(f"{done} out of {total}, bad runs: {badruns_counts}")
    elif mode == 'parallel':
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    solve_instance,
                    coef23,
                    duration,
                    t_dev,
                    n_coef23,
                    dicsr_points,
                    mu_max,
                    t_tar,
                    remove_mu_constr,
                    dam_fun_Weitzman,
                    mu_incr,
                )
                for coef23 in range(n_coef23)
            ]
            for future in futures:
                coef23, q_val, tat_val, success = future.result()
                if success:
                    Xboundary[coef23] = q_val
                    Yboundary[coef23] = tat_val
                else:
                    badruns_counts += 1
                if (coef23+1) % 20 == 0:
                    print(f"{coef23+1} out of {n_coef23}, bad runs: {badruns_counts}")
    elif mode == 'ray':
        i=0
        model_initialized = False
        for coef2 in [-1, 1]:
            for r_i in range(n_rays):
                if not model_initialized:
                    m = ConcreteModel()
                    m.t = ContinuousSet(bounds=(0, duration), initialize = dicsr_points)
                    ### Declare all suffixes
                    # Ipopt bound multipliers (obtained from solution)
                    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                    # Ipopt bound multipliers (sent to solver)
                    m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                    m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
                    # Obtain dual solutions from first solve and send to warm start
                    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

                    m = ncc_rs.makeNCCModel(
                        m,
                        model_mode='RS_mod_util_ray',
                        remove_mu_constr=remove_mu_constr,
                        dam_fun_Weitzman=dam_fun_Weitzman,
                        mu_incr=mu_incr,
                    )
                    for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(m.mu0)
                    model_initialized = True

                iter_counts += 1
                m.coef2 = coef2
                # print(bounds, n_rays)
                if coef2>0:
                    m.coef1 = 0
                    m.coef3 = np.tan(np.pi/(n_rays+2)*(r_i+1))/bounds[3]*bounds[1]
                else:
                    m.coef1 = (1- np.tan(np.pi/(n_rays+2)*(r_i+1)))*bounds[1]
                    m.coef3 = np.tan(np.pi/(n_rays+2)*(r_i+1))/bounds[3]*bounds[1]

                # if coef2>0:
                # 	m.coef1 = bounds[0] - 0.05
                # 	m.coef3 = sin(0.5*np.pi/(n_rays+1)*(r_i+1))/bounds[3]
                # else:
                # 	m.coef1 = bounds[1] + 0.05
                # 	m.coef3 = sin(0.5*np.pi/(n_rays+1)*(r_i+1))/bounds[3]
                #
                solver=SolverFactory('ipopt') # bonmin
                solver.options['halt_on_ampl_error'] = 'yes'

                solver.options['acceptable_tol'] = 1e-4
                solver.options['constr_viol_tol'] = 1e-8
                solver.options['max_iter'] = 700

                try:
                    results = solver.solve(m,tee=True)

                    if (results.solver.status == pyomo.opt.SolverStatus.ok) and (results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal):
                        print('SOLVED OK t_dev = ' + str(t_dev) + '; iter '+ str(iter_counts) + ' out of total ' +str(n_rays*2) + ' ( br ' + str(badruns_counts)+')' )
                    else:
                        badruns_counts += 1
                        print('=( not solved t_dev = ' + str(t_dev) + '; br ' + str(badruns_counts) + '; iter '+ str(iter_counts) + ' out of total ' +str(n_rays*2))
                        continue
                        # scale_dic[exper] = [-1]
                except Exception:
                    continue
                # Xboundary[(coef1, coef23)] = value(m.K[t_tar])
                Xboundary[(coef2, r_i)] = value(m.Q[t_tar])
                Yboundary[(coef2, r_i)] = value(m.TAT_IPCC[t_tar])
                # Xboundary[(coef1, coef23)] = value(m.Ychange_av)
                # Yboundary[(coef1, coef23)] = value(m.TATMav)
                # Zboundary[(coef1, coef23)] = value(m.UTILITY)

                # # except:
                # 	if BAU_RUN == True:
                # 		test_var_duals(m, thr = 0.001)
                # 		sys.exit()
                #
                # 	# continue


    #Save dataframes
    list_keys = list(Yboundary.keys())
    X = np.asarray([ Xboundary[i] for i in list_keys])
    Y = np.asarray([ Yboundary[i] for i in list_keys])
    return X, Y

def solve_instance(
    coef23,
    duration,
    t_dev,
    n_coef23,
    dicsr_points,
    mu_max=1.2,
    t_tar=17,
    remove_mu_constr=False,
    dam_fun_Weitzman=False,
    mu_incr=1.1,
):
    try:
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, duration), initialize=dicsr_points)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        m = ncc_rs.makeNCCModel(
            m,
            model_mode='RS_mod_util',
            tt=t_tar,
            remove_mu_constr=remove_mu_constr,
            dam_fun_Weitzman=dam_fun_Weitzman,
            mu_incr=mu_incr,
        )

        for i in m.t:
            if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                m.mu[i].fix(m.mu0)

        m.coef1 = 0 # 
        m.coef2 = np.sin(2 * np.pi * coef23 / float(n_coef23))
        m.coef3 = np.cos(2 * np.pi * coef23 / float(n_coef23)) / 250
        m.mu_max = mu_max

        solver = SolverFactory('ipopt')
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['acceptable_tol'] = 1e-4
        solver.options['constr_viol_tol'] = 1e-8
        solver.options['max_iter'] = 7000

        results = solver.solve(m, tee=False)

        if results.solver.status == pyomo.opt.SolverStatus.ok and \
           results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
            return coef23, value(m.Q[t_tar]), value(m.TAT_IPCC[t_tar]), True
        else:
            return coef23, None, None, False

    except Exception as e:
        return coef23, None, None, False

def solve_instance_grid(
    util_control,
    delta,
    bounds,
    t_dev,
    duration,
    dicsr_points,
    n_rays,
    mu_max=1.2,
    t_tar=17,
    remove_mu_constr=False,
    dam_fun_Weitzman=False,
    mu_incr=1.1,
):
    try:
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, duration), initialize=dicsr_points)

        # Set suffixes for IPOPT
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # Build model
        m = ncc_rs.makeNCCModel(
            m,
            model_mode='RS_mod_util',
            tt=t_tar,
            remove_mu_constr=remove_mu_constr,
            dam_fun_Weitzman=dam_fun_Weitzman,
            mu_incr=mu_incr,
        )

        for i in m.t:
            if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                m.mu[i].fix(m.mu0)

        m.coef1 = 0
        m.coef2 = util_control[0]
        m.coef3 = util_control[1]
        m.mu_max = mu_max

        if util_control[0] == 0:
            m.TAT_IPCC[t_tar].fix(bounds[0] + delta * (bounds[1] - bounds[0]) / n_rays)
            m.Q[t_tar].unfix()
        else:
            m.Q[t_tar].fix(bounds[2] + delta * (bounds[3] - bounds[2]) / n_rays)
            m.TAT_IPCC[t_tar].unfix()

        solver = SolverFactory('ipopt')
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['acceptable_tol'] = 1e-6
        solver.options['constr_viol_tol'] = 1e-9
        solver.options['max_iter'] = 700

        results = solver.solve(m, tee=False)

        if (results.solver.status == SolverStatus.ok and 
            results.solver.termination_condition == TerminationCondition.optimal):
            return (util_control, delta, value(m.Q[t_tar]), value(m.TAT_IPCC[t_tar]), True)
        else:
            return (util_control, delta, None, None, False)

    except Exception as e:
        print(f"Failed: {e}")
        return (util_control, delta, None, None, False)


def solve_instance_grid_sweep(
    util_control,
    bounds,
    t_dev,
    duration,
    dicsr_points,
    n_rays,
    mu_max=1.2,
    t_tar=17,
    remove_mu_constr=False,
    dam_fun_Weitzman=False,
    mu_incr=1.1,
):
    sweep_results = []
    try:
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, duration), initialize=dicsr_points)

        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        m = ncc_rs.makeNCCModel(
            m,
            model_mode='RS_mod_util',
            tt=t_tar,
            remove_mu_constr=remove_mu_constr,
            dam_fun_Weitzman=dam_fun_Weitzman,
            mu_incr=mu_incr,
        )

        for i in m.t:
            if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                m.mu[i].fix(m.mu0)

        m.coef1 = 0
        m.coef2 = util_control[0]
        m.coef3 = util_control[1]
        m.mu_max = mu_max

        solver = SolverFactory('ipopt')
        solver.options['halt_on_ampl_error'] = 'yes'
        solver.options['acceptable_tol'] = 1e-6
        solver.options['constr_viol_tol'] = 1e-9
        solver.options['max_iter'] = 500
        solver.options['warm_start_init_point'] = 'yes'
        solver.options['print_level'] = 0

        center = n_rays // 2
        center_frac = 0.5 if n_rays == 0 else float(center) / float(n_rays)
        delta_order = sorted(range(n_rays + 1), key=lambda d: abs(d - center))
        eps = 1e-3

        for delta in delta_order:
            frac = 0.5 if n_rays == 0 else float(delta) / float(n_rays)
            # Avoid exact endpoints, which are often numerically unstable/infeasible.
            frac = min(max(frac, eps), 1.0 - eps)

            if util_control[0] == 0:
                target = bounds[0] + frac * (bounds[1] - bounds[0])
                m.TAT_IPCC[t_tar].fix(target)
                m.Q[t_tar].unfix()
            else:
                target = bounds[2] + frac * (bounds[3] - bounds[2])
                m.Q[t_tar].fix(target)
                m.TAT_IPCC[t_tar].unfix()

            solve_ok = False
            for retry_idx, max_iter in enumerate((500, 900)):
                solver.options['max_iter'] = max_iter
                try:
                    results = solver.solve(m, tee=False, load_solutions=False)
                    if (results.solver.status == SolverStatus.ok and
                        results.solver.termination_condition == TerminationCondition.optimal):
                        m.solutions.load_from(results)
                        sweep_results.append((delta, value(m.Q[t_tar]), value(m.TAT_IPCC[t_tar]), True))
                        solve_ok = True
                        break
                    # One retry for common local failures at endpoints/max-iter.
                    if retry_idx == 0:
                        retry_frac = 0.5 * frac + 0.5 * center_frac
                        if util_control[0] == 0:
                            m.TAT_IPCC[t_tar].fix(bounds[0] + retry_frac * (bounds[1] - bounds[0]))
                        else:
                            m.Q[t_tar].fix(bounds[2] + retry_frac * (bounds[3] - bounds[2]))
                except ApplicationError:
                    if retry_idx == 0:
                        continue
                except Exception:
                    if retry_idx == 0:
                        continue
            if not solve_ok:
                sweep_results.append((delta, None, None, False))

        return util_control, sweep_results

    except Exception:
        for delta in range(n_rays + 1):
            sweep_results.append((delta, None, None, False))
        return util_control, sweep_results

