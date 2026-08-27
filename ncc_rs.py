# This code is an implementation of AMPL code of H�nsel et.al (2020): "Climate economics support for the UN climate targets"
# that works in pyomo + python + IPOPT setting
# We follow closely the original code, hence we use quotations

from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
import scipy
import numpy as np
def makeNCCModel(
    m,
    model_mode = 'vanilla',
    tt = 17,
    remove_mu_constr = False,
    dam_fun_Weitzman = False,
    mu_incr = 1.1,
):
    
    # model_mode = 'vanilla' -- run the original model from the NCC paper
    # model_mode = 'RS_grid' -- using a grid to approximate RS
    # model_mode = 'RS_mod_util' -- using a modified utility to approximate RS
    # model_mode = 'RS_mod_util_ray' -- using a modified utility to approximate RS
    # remove_mu_constr = False -- the original constraint is left
    # remove_mu_constr = True -- we remove m.mu[i] <= m.mu[prt]*1.1 that starts from t=7
    # dam_fun_Weitzman = True -- we use damagage function of Weitzman


    m.mu_max = Param(initialize=1.2, mutable=True) # variable negative emissions
    m.mu_incr = Param(initialize=mu_incr, mutable=True) # post-2160 growth factor for mu

    # Preferences
    # "inequality aversion: under Nordhaus optimal policy substitute by: param eta:=1.45; for median expert: param eta:=1.0000001;"
    m.eta = Param(initialize = 1.45)
    # "pure time preference: under Nordhaus optimal policy substitute by: param rho:=0.015; for median expert:param rho:=0.005;"
    m.rho = Param(initialize = 0.015)
    # discount factor
    def discount_factor(m, t):
        return 1./((1.+m.rho)**(5*t))
    m.R = Param(m.t, initialize = discount_factor)

    # "Population and its dynamics, including assumption on asymptotic population of 11500 millions"
    m.L0 = Param(initialize = 7403)                 #"initial world population 2015 (millions)"
    m.gL0 = Param(initialize = 0.134)                 #"growth rate to calibrate to 2050 pop projection"
    def pop_dyn(m,t):
        if t == m.t.first():
            return m.L0
        else:
            prt = m.t.prev(t)
            return  m.L[prt]*((11500/m.L[prt])**m.gL0)
    m.L = Param(m.t, initialize = pop_dyn)

    # "Technology and its dynamics"
    m.gamma = Param(initialize = 0.3)                  #"capital elasticity in production function"
    m.deltaK = Param(initialize = 0.100)             #"depreciation rate on capital (per year)"
    m.Qgross0 = Param(initialize = 105.5)             #"Initial world gross output 2015 (trill 2010 USD)"
    m.K0 = Param(initialize = 223)                     #"initial capital value 2015 (trillions 2010 USD)"
    m.A0 = Param(initialize = 5.115)                 #"initial level of total factor productivity"
    m.gA0 = Param(initialize = 0.076)                 #"initial growth rate for TFP per 5 years"
    m.deltaA = Param(initialize =0.005)             #"decline rate of TFP per 5 years"
    def tech_dyn1(m,t):
            return  m.gA0*exp(-m.deltaA*5*t)
    m.gA = Param(m.t, initialize = tech_dyn1)

    def tech_dyn2(m,t):
        if t == m.t.first():
            return m.A0
        else:
            prt = m.t.prev(t)
            return  m.A[prt]/(1-m.gA[prt])
    m.A = Param(m.t, initialize = tech_dyn2)

    #"Emission parameters, where sigma is the carbon intensity or CO2-output ratio"
    m.gsigma0 = Param(initialize =-0.0152)    #"initial growth of sigma (coninuous per year )"
    m.deltasigma = Param(initialize =-0.001) #"decline rate of decarbonization per period"
    m.ELand0 = Param(initialize =2.6) #"initial Carbon emissions from land 2015 (GtCO2 per period)"
    m.deltaLand = Param(initialize =0.115) #"decline rate of land emissions (per period)"
    m.EInd0 = Param(initialize =35.85) #"industrial emissions 2015 (GtCO2 per year)"
    m.Ecum0 = Param(initialize =400) #"initial cumulative emissions (GtCO2)"
    m.mu0 = Param(initialize =.03) #"initial emissions control rate for base year 2010; under BAU: 0.00"
    # m.Lambda0 = Param(initialize =0) #"initial abatement costs" (not used in the code)
    m.sigma0 = Param(initialize = m.EInd0/(m.Qgross0*(1-m.mu0)))#"initial sigma (kgCO2 per output)"

    def gsigma_dyn(m,t):
        if t == m.t.first():
            return m.gsigma0
        else:
            prt = m.t.prev(t)
            return  m.gsigma[prt]*((1+m.deltasigma)**5)
    m.gsigma = Param(m.t, initialize = gsigma_dyn)

    def sigma_dyn(m,t):
        if t == m.t.first():
            return m.sigma0
        else:
            prt = m.t.prev(t)
            return  m.sigma[prt]*exp(m.gsigma[prt]*5)
    m.sigma = Param(m.t, initialize = sigma_dyn)

    def ELand_dyn(m,t):
        if t == m.t.first():
            return m.ELand0
        else:
            prt = m.t.prev(t)
            return  m.ELand[prt]*(1-m.deltaLand)
    m.ELand = Param(m.t, initialize = ELand_dyn)

    def CumLand_dyn(m,t):
        if t == m.t.first():
            return 197.0
        else:
            prt = m.t.prev(t)
            return  m.CumLand[prt] + m.ELand[prt]*(5/3.666)
    m.CumLand = Param(m.t, initialize = CumLand_dyn)

    # "Carbon cycle"
    m.MAT0 = Param(initialize = 127.159+93.313+37.840+7.721+588.000) # "Initial Concentration in atmosphere 2015 (GtC)""
    m.MATEQ = Param(initialize = 588)                 # "Equilibrium concentration in atmosphere   (GtC)""
    m.MUPEQ = Param(initialize = 360)                 # "Equilibrium concentration in upper strata (GtC)""
    m.MLOEQ = Param(initialize = 1720)                 # "Equilibrium concentration in lower strata (GtC)""

    # "Impulse response according to IPCC AR5"
    m.box = Set(initialize = [1,2,3,4])
    t_scale_data = {1: 1000000, 2: 394.4, 3: 36.54, 4: 4.304}
    m.t_scale = Param(m.box, initialize = t_scale_data)
    fraction_data = {1: 0.217, 2: 0.224, 3: 0.282, 4: 0.276}
    m.fraction = Param(m.box, initialize = fraction_data)

    # "Climate model parameters"
    m.nu = Param(initialize =3.1, mutable = True) # "Equilibrium temperature impact (�C per doubling C02)"
    m.TLO0 = Param(initialize =0.324) # "Initial lower stratum temperature change (�C from 1900) # adjusted to only include athropogenic forcing"
    m.TAT0 = Param(initialize =1.243) # "Initial atmospheric temp change (�C from 1900) # adjusted to only include athropogenic forcing"
    m.delta_T = Param(initialize =0.115) # "adjustment to compare to 1850-1900 temperature levels"
    m.xi1 = Param(initialize =7.3) # "Speed of adjustment m. = Parameter for atmospheric temperature"
    m.xi3 = Param(initialize =0.73) # "Coefficient of heat loss from atmosphere to oceans"
    m.xi4 = Param(initialize =106)    # "Coefficient of heat gain by deep oceans"
    m.kappa = Param(initialize =3.6813) # "Forcings of equilibrium CO2 doubling (Wm-2)"
    m.xi2 = Param(initialize = m.kappa/m.nu, mutable = True) # "climate model parameter"

    FexDF = pd.read_csv('diceParams/nonCO2_forcing.csv')
    fex = dict(zip(FexDF.period, FexDF.value))
    def _ExogenousForcingOfOtherGreenhouseGases(model, i):
        return fex[i]
    m.Fex = Param(m.t, initialize=_ExogenousForcingOfOtherGreenhouseGases)

    # # climate damage parameters
    m.Psi = Param(initialize =0.007438)             # "Based on Howard and Sterner (2017)"
    m.TATlim = Param(initialize =5+0.113)            # "upper bound on atm. temperature change"
    #
    # # abatement cost
    m.Theta = Param(initialize =2.6)                 # "Exponent of control cost function"
    m.pback0 = Param(initialize =550)                 # "Cost of backstop 2010 $ per tCO2 2015"
    m.gback = Param(initialize =0.025)             # "Initial cost decline backstop cost per period"
    m.cprice0 = Param(initialize =2)                 # "Initial base carbon price (2010$ per tC02)"
    #
    def pback_dyn(m,t):
        if t == m.t.first():
            return m.pback0
        else:
            prt = m.t.prev(t)
            return  m.pback[prt]*(1-m.gback)
    m.pback = Param(m.t, initialize = pback_dyn)
    def phead_dyn(m,t):
            return m.pback[t]*m.sigma[t]/m.Theta/1000
    m.phead = Param(m.t, initialize = phead_dyn)

    # # VARIABLES
    #
    # capital (trillions 2010 USD)
    m.K = Var(m.t, bounds = (1, np.inf))

    # Gross output (trillions 2010 USD)
    m.Qgross = Var(m.t, initialize = 1)
    def Gross_output_rule(m, t):
        return m.Qgross[t] == m.A[t]*((m.L[t]/1000.)**(1-m.gamma))*(m.K[t]**m.gamma)
    m.Gross_output = Constraint(m.t, rule = Gross_output_rule)

    # carbon cycle
    # "carbon reservoir atmosphere (GtC)"
    m.MAT = Var(m.t,  bounds = (10., np.inf))
    #
    m.alpha = Var(m.t, domain = NonNegativeReals, initialize = 0.1)#  bounds = (0.001, 100))
    #
    m.c_cycle = Var(m.t, m.box )

    m.F = Var(m.t)
    def forcing_rule(m, t):
        return m.F[t] == m.kappa*((log(m.MAT[t]/m.MATEQ))/log(2))+m.Fex[t]
    m.forcing = Constraint(m.t, rule = forcing_rule)

    # # atmospheric temperature change (�C from 1750)
    m.TAT = Var(m.t,  bounds = (0, m.TATlim))
    m.ts = Set(initialize = [1,2,3,4,5])
    # # atmospheric temperature change short (�C from 1750)
    m.TAT_short = Var(m.t,  m.ts, domain = NonNegativeReals)

    # # ocean temperature (�C from 1750)
    m.TLO = Var(m.t, bounds = (-1, 20))
    #
    # # atmospheric temperature change (�C from 1850-1900)
    m.TAT_IPCC = Var(m.t, domain = NonNegativeReals, initialize = 2.0)
    def TAT_IPCC_rule(m, t):
        return m.TAT_IPCC[t] == m.TAT[t]- m.delta_T
    m.TAT_IPCC_const = Constraint(m.t, rule = TAT_IPCC_rule)

    # # damage fraction
    m.Omega = Var(m.t)
    def damage_fractionrule(m, t):
        if dam_fun_Weitzman == True:
            return m.Omega[t] == 1-(1/((1+(m.TAT[t]/20.5847)**2)+(m.TAT[t]/6.081)**(6.754)))
        else:
            return m.Omega[t] == m.Psi*(m.TAT_IPCC[t])**2

    m.damage_fraction= Constraint(m.t, rule = damage_fractionrule)
    #
    # # damages (trillions 2010 USD)
    m.damage = Var(m.t)
    def damages_rule(m, t):
        return m.damage[t] == m.Omega[t]*m.Qgross[t]
    m.damages = Constraint(m.t, rule = damages_rule)
    #
    # # emission control
    def miuBounds(model, i):
        if i == model.t.first():
            return (model.mu0, model.mu0)
        elif model.t.ord(i) <= 7: #subject to control1 {t in 2..6}:mu[t]<=1;
            return (0.00001, 1.)
        else:
            return (0.00001, model.mu_max)
    # Emission control rate GHGs
    m.mu = Var(m.t,  bounds = miuBounds)
    #
    # # abatement costs (fraction of output)
    m.Lambda = Var(m.t)
    def abatement_costs_rule(m, t):
        return m.Lambda[t] == m.Qgross[t]*m.phead[t]*(m.mu[t]**m.Theta)
    m.abatement_costs = Constraint(m.t, rule = abatement_costs_rule)


    # # industrial emissions
    m.EInd = Var(m.t)
    def industrial_emissions_rule(m, t):
        return m.EInd[t] == m.sigma[t]*m.Qgross[t]*(1-m.mu[t])
    m.industrial_emissions = Constraint(m.t, rule = industrial_emissions_rule)

    # # total emissions
    m.E = Var(m.t)
    # # maximum cumulative extraction fossil fuels (GtC)
    m.Ecum = Var(m.t, bounds = (-6000, 6000))
    m.cprice = Var(m.t)
    # # Marginal cost of abatement (carbon price)
    def Marginal_cost_rule(m, t):
        return m.cprice[t] == m.pback[t]*m.mu[t]**(m.Theta-1)
    m.Marginal_cost = Constraint(m.t, rule = Marginal_cost_rule)

    # # output net of damages and abatement (trillions 2010 USD)
    m.Q = Var(m.t)
    def output_net_rule(m, t):
        return m.Q[t] ==(m.Qgross[t]*(1-m.Omega[t]))-m.Lambda[t]
    m.output_net = Constraint(m.t, rule = output_net_rule)

    #
    # # per capita consumption (1000s 2010 USD]
    m.c = Var(m.t, bounds = (0.1, np.inf))
    #
    # # aggregate consumption
    m.C = Var(m.t)
    def aggregate_consumption_rule(m, t):
        return m.C[t] == m.L[t]*m.c[t]/1000
    m.aggregate_consumption = Constraint(m.t, rule = aggregate_consumption_rule)

    # # Investment (trillions 2005 USD)
    m.I = Var(m.t, domain = NonNegativeReals)

    #
    # # utility
    m.U = Var(m.t)
    def utility_rule(m, t):
        return m.U[t] == m.c[t]**(1-m.eta)/(1-m.eta)
    m.utility_ = Constraint(m.t, rule = utility_rule)
    #
    # # total period utility
    m.U_period = Var(m.t)
    def total_period_utility_rule(m, t):
        return m.U_period[t] == m.U[t]*m.R[t]
    m.total_period_utility = Constraint(m.t, rule = total_period_utility_rule)

    # # welfare/objective function
    m.W = Var()
    def welfare_rule(m):
        return m.W == sum(m.L[t]*m.U[t]*m.R[t] for t in m.t)
    m.welfare = Constraint(rule = welfare_rule)

    # # welfare optimization
    # maximize objective_function: W;
    def constr_accounting_rule(m, t):
        return m.C[t] == m.Q[t]-m.I[t]
    m.constr_accounting= Constraint(m.t, rule = constr_accounting_rule)
    def constr_emissions_rule(m, t):
        return m.E[t]== m.EInd[t]+m.ELand[t]
    m.constr_emissions = Constraint(m.t, rule = constr_emissions_rule)
    def constr_capital_dynamics_rule(m, t):
        if t == m.t.first():
            return Constraint.Skip
        else:
            prt = m.t.prev(t)
            return m.K[t]== (1-m.deltaK)**5*m.K[prt]+5*m.I[prt]
    m.constr_capital_dynamics = Constraint(m.t, rule = constr_capital_dynamics_rule)
    def constr_cumulativeemissions_rule(m, t):
        if t == m.t.first():
            return Constraint.Skip
        else:
            prt = m.t.prev(t)
            return m.Ecum[t]==m.Ecum[prt]+((m.E[prt]-m.ELand[prt])*5/3.666)
    m.constr_cumulativeemissions = Constraint(m.t, rule = constr_cumulativeemissions_rule)

    def _alpha_calibration(m, t): #только этот менял
        expr1 = sum(m.alpha[t]*m.fraction[box]*m.t_scale[box]*(1-exp(-100./(m.alpha[t] * m.t_scale[box]))) for box in m.box)
        return 35+0.019*((m.Ecum[t]+m.CumLand[t])-(m.MAT[t]-588)) + 4.165*m.TAT[t] == expr1
    m.alpha_calibration = Constraint(m.t, rule=_alpha_calibration)

    def _carbon_cycle_calibration(m, t, box):
        if t == m.t.first():
            return Constraint.Skip
        prt = m.t.prev(t)
        expr1 = m.c_cycle[prt,box]*exp(-5./(m.alpha[prt]*m.t_scale[box]))
        expr2 = m.fraction[box] * (m.E[prt]*exp(-5    /(m.alpha[prt]*m.t_scale[box]))*(1/3.666))
        expr3 = m.fraction[box] * (m.E[prt]*exp(-(5-1)/(m.alpha[prt]*m.t_scale[box]))*(1/3.666))
        expr4 = m.fraction[box] * (m.E[prt]*exp(-(5-2)/(m.alpha[prt]*m.t_scale[box]))*(1/3.666))
        expr5 = m.fraction[box] * (m.E[prt]*exp(-(5-3)/(m.alpha[prt]*m.t_scale[box]))*(1/3.666))
        expr6 = m.fraction[box] * (m.E[prt]*exp(-(5-4)/(m.alpha[prt]*m.t_scale[box]))*(1/3.666))
        return m.c_cycle[t,box] == expr1 + expr2 + expr3 +  expr4 + expr5 + expr6
    m.carbon_cycle_calibration = Constraint(m.t, m.box,rule=_carbon_cycle_calibration)

    def _constr_atmosphere(m, t):
        return m.MAT[t] == sum(m.c_cycle[t,box]  for box in m.box) + 588.
    m.constr_atmosphere = Constraint(m.t, rule=_constr_atmosphere)


    def _constr_atmospheric_temp_1(m, t):
        return m.TAT_short[t,1] == m.TAT[t]
    m.constr_atmospheric_temp_1 = Constraint(m.t, rule=_constr_atmospheric_temp_1)
    def _constr_atmospheric_temp_2(m, t, ts):
        if t == m.t.last():
            return Constraint.Skip
        if ts == m.ts.last():
            return Constraint.Skip
        nxt = m.t.next(t)
        return m.TAT_short[t,ts+1] == m.TAT_short[t,ts] + 1./m.xi1*((m.F[nxt]-m.xi2*m.TAT_short[t,ts])-(m.xi3*(m.TAT_short[t,ts]-m.TLO[t])))
    m.constr_atmospheric_temp_2 = Constraint(m.t, m.ts, rule=_constr_atmospheric_temp_2)
    def _constr_atmospheric_temp_3(m, t):
        if t == m.t.last():
            return Constraint.Skip
        else:
            nxt = m.t.next(t)
            return m.TAT_short[t,5] == m.TAT[nxt]
    m.constr_atmospheric_temp_3 = Constraint(m.t, rule=_constr_atmospheric_temp_3)
    def _constr_ocean_temp(m, t):
        if t == m.t.last():
            return Constraint.Skip
        else:
            nxt = m.t.next(t)
            return m.TLO[nxt] == m.TLO[t]+5*m.xi3/m.xi4*(m.TAT[t]-m.TLO[t])
    m.constr_ocean_temp = Constraint(m.t, rule=_constr_ocean_temp)
    # #Equation below have been rewritten to match a geophysical interpretation of the energy balance model based on Geoffroy (2013)
    #
    #
    #
    # subject to constr_atmospheric_temp_1 {t in 0..T, ts in 1..5}:         TAT_short[t,1]=TAT[t];
    # subject to constr_atmospheric_temp_2 {t in 0..T-1, ts in 1..4}:       TAT_short[t,ts+1]=TAT_short[t,ts] + 1/xi1*((F[t+1]-xi2*TAT_short[t,ts])-(xi3*(TAT_short[t,ts]-TLO[t])));
    # subject to constr_atmospheric_temp_3 {t in 0..T-1, ts in 1..5}:     TAT[t+1]=TAT_short[t,5];
    # subject to constr_ocean_temp {t in 0..T-1}:                         TLO[t+1]=TLO[t]+5*xi3/xi4*(TAT[t]-TLO[t]);
    #
    #

    # # Initial conditions
    m.K[0].fix(m.K0)
    m.Ecum[0].fix(m.Ecum0)
    m.MAT[0].fix(m.MAT0)
    m.TLO[0].fix(m.TLO0)
    m.TAT[0].fix(m.TAT0)
    # m.mu[0].fix(m.mu0) #not needed

    def _control1a_3(m, i):
        if m.t.ord(i) == 1:
            return Constraint.Skip
        if m.t.ord(i) == 2:
            return m.mu[i] == 1-(m.EInd0*(1.01)**5)/(m.sigma[i]*m.Qgross[i])
        prt = m.t.prev(i)
        if m.t.ord(i) <= 7:
            return m.mu[i] <= 1-(m.EInd[prt]-10)/(m.sigma[i]*m.Qgross[i])
        else:
            if remove_mu_constr == False:
                return m.mu[i] <= m.mu[prt]*m.mu_incr
                # return Constraint.Skip
            else:
                return Constraint.Skip
    m.control1a_3 = Constraint(m.t, rule= _control1a_3)
    # subject to initial_control_2020:     mu[1]=1-(EInd0*(1.01)^5)/(sigma[1]*Qgross[1]);
    # subject to control1a {t in 2..6}:     mu[t]<=1-(EInd[t-1]-10)/(sigma[t]*Qgross[t]);
    # subject to control1 {t in 2..6}:     mu[t]<=1;
    # subject to control2 {t in 7..T}:     mu[t]<=1.2;        # from 2160
    # subject to control3 {t in 7..T}:     mu[t]<=mu[t-1]*mu_incr;
    #
    #
    m.c_cycle[0,1].fix(127.159)
    m.c_cycle[0,2].fix(93.313)
    m.c_cycle[0,3].fix(37.840)
    m.c_cycle[0,4].fix(7.721)
    # m.UTILITY = Var(domain=Reals, bounds = (0,1))

    m.S = Var(m.t, bounds = (0.0001, 0.3))
    # Savings rate equation; I(t)=E= S(t) * Y(t);
    def _savingsRate(m, t):
        return m.I[t] == m.S[t]*m.Q[t]
    m.savingsRate = Constraint(m.t, rule=_savingsRate)
    
    # def _S_constr1(m, t):
    #     return inequality(0.15 * (m.Q[t]+0.00001), m.I[t], 0.3 * (m.Q[t]+0.00001))
    # m.S_constr1 = Constraint(m.t, rule = _S_constr1)

    if model_mode == 'RS_mod_util':
        ### only to produce RS#######################
        m.coef1 = Param(initialize=1.0, mutable=True)
        m.coef2 = Param(initialize=1.0, mutable=True)
        m.coef3 = Param(initialize=1.0, mutable=True)
        def obj_rule(m):
            #TAT_IPCC atmospheric temperature change (�C from 1850-1900)
            return  m.coef1*m.W + m.coef2*m.TAT_IPCC[tt] + m.coef3*m.Q[tt] #we focused on RS in 2100
        #############################################
    elif model_mode == 'vanilla':
        def obj_rule(m):
            return  m.W
    elif model_mode == 'RS_mod_util_ray':
        m.coef1 = Param(initialize=1.0, mutable=True)
        m.coef2 = Param(initialize=1.0, mutable=True)
        m.coef3 = Param(initialize=1.0, mutable=True)
        def obj_rule(m):
            return  m.coef2*m.TAT_IPCC[tt]
        def _RS_ray(model):
            return model.TAT_IPCC[tt] == model.coef1 + model.coef3*model.Q[tt]
        m.RS_ray = Constraint(rule=_RS_ray)

    m.OBJ = Objective(rule=obj_rule, sense=maximize)
    return m
