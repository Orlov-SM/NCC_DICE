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

    # Solve
    solver = SolverFactory("ipopt")
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['acceptable_tol'] = 1e-6
    solver.options['constr_viol_tol'] = 1e-9
    solver.options['max_iter'] = 1000
    results = solver.solve(m, tee=False)

    if (results.solver.status != SolverStatus.ok or
        results.solver.termination_condition != TerminationCondition.optimal):
        print("⚠️ Solver failed or did not reach optimality.")
        return None, None

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
def run_boundary_NCC(t_dev, mu_max, coef_list = extreme_coef_list):
    T_list = []
    Y_list = []
    t_tar = 17
    for coef_list_i in coef_list:
        m = ConcreteModel()
        duration = 20
        dicsr_points = range(20)
        m.t = ContinuousSet(bounds=(0, duration), initialize = dicsr_points)
        m = ncc_rs.makeNCCModel(m, model_mode = 'RS_mod_util')
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
        results = solver.solve(m,tee=True)
        T_list.append(m.TAT_IPCC[17].value)
        Y_list.append(m.Q[17].value)
    # return  T_list, Y_list
    return (min(T_list), max(T_list), min(Y_list), max(Y_list),)


#This function is used to compute a RS for NCC mod of the DICE model (similar to RS_bulk)
#for vanilla case 4.25 >= tat_IPCC in  2100 >= 0.8179; 1450 >= Q in 2100 >= 78
#we use either grid approximation, modified utility or ray
def RS_bulk_ncc(bounds, t_dev, t_tar, remove_mu_constr, dam_fun_Weitzman, resol = (32,10,), mode = 'mod_util', mu_max = 1.2, opt = False, mu_opt = np.arange(20)): #'grid', 'mod_util', 'ray'
    # #here we define 500 years
    # duration = 99 # 300
    # tttstep = 1 #5.0
    # dicsr_points = range(100) #np.arange(0.0, duration, tttstep)
    #but actually we need only 100 years and less
    duration = t_tar+1 # 300
    tttstep = 1 #5.0
    dicsr_points = range(duration) #np.arange(0.0, duration, tttstep)


    first_run = True
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
       if first_run:
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
            m = ncc_rs.makeNCCModel(m, model_mode = 'RS_mod_util', tt=t_tar)
            if opt:
                for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(mu_opt[i])
            else:
                for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(m.mu0)
            first_run = False
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
                        results = solver.solve(m,tee=True)
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
        # for coef1 in range(n_coef1):
            # coef1_val = 1 - (coef1 + 1)/float(n_coef1)
        for coef23 in range(n_coef23):
            # try:
            if first_run:
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

                m = ncc_rs.makeNCCModel(m, model_mode = 'RS_mod_util', tt=t_tar)
                # makeNCCModel(m, model_mode = 'vanilla', remove_mu_constr = False, dam_fun_Weitzman = False)
                for i in m.t:
                    if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                        m.mu[i].fix(m.mu0)
                first_run = False

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
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    solve_instance_grid,
                    uc, delta, bounds, t_dev, duration, dicsr_points, resol, mu_max, t_tar
                )
                for uc in util_controls
                for delta in range(n_rays + 1)
            ]
            for future in futures:
                uc, delta, q_val, tat_val, success = future.result()
                if success:
                    Xboundary[(uc, delta)] = q_val
                    Yboundary[(uc, delta)] = tat_val
                else:
                    badruns_counts += 1
                    print(f"Failed for util_control={uc}, delta={delta}")
                if (delta+1) % 20 == 0:
                    print(f"{coef23+1} out of {n_coef23}, bad runs: {badruns_counts}")
    elif mode == 'parallel':
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(solve_instance, coef23, duration, t_dev, n_coef23, dicsr_points, mu_max, t_tar) for coef23 in range(n_coef23)]
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
        for coef2 in [-1, 1]:
            for r_i in range(n_rays):
                if first_run:
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

                    m = ncc_rs.makeNCCModel(m, model_mode = 'RS_mod_util_ray', remove_mu_constr = remove_mu_constr, dam_fun_Weitzman = dam_fun_Weitzman)
                    for i in m.t:
                        if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                            m.mu[i].fix(m.mu0)
                    first_run = False

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

def solve_instance(coef23, duration, t_dev, n_coef23, dicsr_points, mu_max=1.2, t_tar=17):
    try:
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, duration), initialize=dicsr_points)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        m = ncc_rs.makeNCCModel(m, model_mode='RS_mod_util', tt=t_tar)

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

def solve_instance_grid(util_control, delta, bounds, t_dev, duration, dicsr_points, resol, mu_max=1.2, t_tar=17):
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
        m = ncc_rs.makeNCCModel(m, model_mode='RS_mod_util', tt=t_tar)

        for i in m.t:
            if m.t.ord(i) > 2 and m.t.ord(i) < t_dev:
                m.mu[i].fix(m.mu0)

        m.coef1 = 0
        m.coef2 = util_control[0]
        m.coef3 = util_control[1]
        m.mu_max = mu_max

        if util_control[0] == 0:
            m.TAT_IPCC[t_tar].fix(bounds[0] + delta * (bounds[1] - bounds[0]) / resol)
            m.Q[t_tar].unfix()
        else:
            m.Q[t_tar].fix(bounds[2] + delta * (bounds[3] - bounds[2]) / resol)
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