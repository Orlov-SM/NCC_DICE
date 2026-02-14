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

# import logging

# class CrashOnW1002(logging.Handler):
#     def emit(self, record):
#         if "Setting Var" in record.getMessage() and "outside the bounds" in record.getMessage():
#             raise RuntimeError(record.getMessage())

# logging.getLogger('pyomo').addHandler(CrashOnW1002())
# logging.getLogger('pyomo').setLevel(logging.WARNING)

####################################
## i) This part computes RS at t_tar for the ***NCC paper*** for
## all deviations from the BAU path at specified time instant t_dev
####################################
miu_incr = False #no incremental constraints
dam_fun_Weitzman = False #select the damage function


# fig, axes = plt.subplots(1, 1, dpi=200, figsize = (4,4))
# # lince_col_dic = {1:'y', 5:'r', 9:'b', 13:'g'}
# lince_col_dic = {1:'y', 5:'r', 7:'b', 9:'g'}
# # marker_dic = {1:'o', 5:'<', 9:'>', 13:'o'}
# marker_dic = {1:'o', 5:'<', 7:'>', 9:'o'}

# resol1 =  (1, 150) #defines the level of approximation, higher product of this numbers gives better approximation
resol1 =  (1, 10)
# X= np.asarray([])
# Y= np.asarray([])

t_start = datetime.datetime.now()
# miu_incr = 0.15 #constraint for incremental change of MIU
extreme_res = {}

# Run for changing mu_max
if __name__ == '__main__':
	# list_of_runs = [13, 9, 5, 1]
	# list_of_runs = [9, 7, 5, 1]
	mu_max_range = np.linspace(1.2, 2.21, 5)
	for mu_max in mu_max_range:
		if np.isclose(mu_max, 1.2):
			list_of_runs = [1, 2, 3, 4, 5, 6, 7] 
		else:
			list_of_runs = [2, 3, 4, 5, 6, 7]

		for t_dev in list_of_runs:
			extreme_res[t_dev] = (1, 3.5, 580, 850)
			X_t, Y_t = RSroutines.RS_bulk_ncc(
				extreme_res[t_dev], t_dev, 17,
				remove_mu_constr=miu_incr,
				dam_fun_Weitzman=dam_fun_Weitzman,
				resol=resol1, mode='grid',
				mu_max=mu_max  # <- pass this down
			)

			points = np.zeros([len(X_t), 2])
			for k in range(len(X_t)):
				points[k, 0] = X_t[k]
				points[k, 1] = Y_t[k]

			hull = scipy.spatial.ConvexHull(points)
			filename = f'plots_data/NCC_mu{mu_max:.2f}_tdev{t_dev}.csv'
			np.savetxt(filename, points[hull.vertices, :], delimiter=',')
	print("Total time:", datetime.datetime.now() - t_start)

# # Run for changing t_tar
# if __name__ == '__main__':
# 	mu_max = 1.2
# 	t_tar_range = np.arange(17, 20, 1)
# 	for t_tar in t_tar_range:
# 		if np.isclose(t_tar, 17):
# 			list_of_runs = [1, 2, 3, 4, 5, 6, 7]
# 		else:
# 			list_of_runs = [2, 3, 4, 5, 6, 7]

# 		for t_dev in list_of_runs:
# 			extreme_res[t_dev] = (1, 3.5, 580, 850)
# 			X_t, Y_t = RSroutines.RS_bulk_ncc(
# 				extreme_res[t_dev], t_dev, t_tar,
# 				remove_mu_constr=miu_incr,
# 				dam_fun_Weitzman=dam_fun_Weitzman,
# 				resol=resol1, mode='grid',
# 				mu_max=mu_max, opt = False  # <- pass this down
# 			)

# 			points = np.zeros([len(X_t), 2])
# 			for k in range(len(X_t)):
# 				points[k, 0] = X_t[k]
# 				points[k, 1] = Y_t[k]

# 			hull = scipy.spatial.ConvexHull(points)
# 			filename = f'plots_data/NCC_t_tar{t_tar:.2f}_tdev{t_dev}.csv'
# 			np.savetxt(filename, points[hull.vertices, :], delimiter=',')
# 	print("Total time:", datetime.datetime.now() - t_start)


# t, mu_opt, S = RSroutines.solve_vanilla_optimal_path()

# # Run for changing mu_max for optimal mu
# if __name__ == '__main__':
# 	mu_max_range = np.arange(1.2, 0.99, -0.05)
# 	t_tar = 17
# 	for mu_max in mu_max_range:
# 		if np.isclose(mu_max, 1.2):
# 			list_of_runs = [1, 2, 3, 4, 5, 6, 7]
# 		else:
# 			list_of_runs = [2, 3, 4, 5, 6, 7]
# 		for t_dev in list_of_runs:
# 			extreme_res[t_dev] = (1, 3.5, 580, 850)
# 			X_t, Y_t = RSroutines.RS_bulk_ncc(
# 				extreme_res[t_dev], t_dev, t_tar,
# 				remove_mu_constr=miu_incr,
# 				dam_fun_Weitzman=dam_fun_Weitzman,
# 				resol=resol1, mode='grid',
# 				mu_max=mu_max, opt = 'True', mu_opt=mu_opt  # <- pass this down
# 			)
# 			points = np.zeros([len(X_t), 2])
# 			for k in range(len(X_t)):
# 				points[k, 0] = X_t[k]
# 				points[k, 1] = Y_t[k]
# 			hull = scipy.spatial.ConvexHull(points)
# 			filename = f'plots_data/NCC_mu{mu_max:.2f}_tdev{t_dev}_mu_opt.csv'
# 			np.savetxt(filename, points[hull.vertices, :], delimiter=',')
# 	print("Total time:", datetime.datetime.now() - t_start)


	# list_of_runs = [9, 7, 5]
	# for t_dev in list_of_runs:
	# 	#compute and save extreme points of RS
	# 	# extreme_res[t_dev] = run_boundary(t_dev, miu_incr = miu_incr)
	# 	# (min(T_list), max(T_list), min(Y_list), max(Y_list),)
	# 	extreme_res[t_dev]  = (0.8, 4.3, 78, 1450)
	# 	#compute and save RS
	# 	X_t, Y_t = RSroutines.RS_bulk_ncc(extreme_res[t_dev], t_dev, remove_mu_constr = miu_incr, dam_fun_Weitzman = dam_fun_Weitzman,  resol = resol1, mode = 'parallel')
	# 	# X = np.hstack((X, X_t))
	# 	# Y = np.hstack((Y, Y_t))
	# 	# print(X)
	# 	points = np.zeros([len(X_t), 2])

	# 	for k in range(len(X_t)):
	# 		points[k,0] = X_t[k]
	# 		points[k,1] = Y_t[k]
	# 		plt.plot(X_t[k],Y_t[k], color = lince_col_dic[t_dev], marker=marker_dic[t_dev], markersize=1)
	# 	hull = scipy.spatial.ConvexHull(points)
	# 	# folder_name = 'NCC_DICE_miu_incr'+ str(miu_incr)+ '_dam_fun'+ str(dam_fun_Weitzman)+"/"
	# 	np.savetxt('plots_data/NCC_DICE_plots_data/' + str(t_dev)+ ".csv", points[hull.vertices, ], delimiter=",")

		# line = plt.Polygon(points[hull.vertices, ], fill=None, edgecolor= lince_col_dic[t_dev],label = str(2015 + 5*t_dev) )
	# 	# # if miu_incr != 'No':
	# 	# # 	# plt.title('ncc_miu_incr run ' + str(miu_incr))
	# 	# # 	plt.title('Negative emissions 1.5')
		# plt.gca().add_patch(line)

	# # save_obj(extreme_res, 'paper_plots_data/extreme_res_for_norm')
	# print (datetime.datetime.now()-t_start)
	# # #plots and saves the draft picture
	# # plt.savefig( 'paper_plots/ncc_RS 13-9-5-1 iter = ' + str(resol1) +'miu_incr'+ str(miu_incr)+ '_dam_fun'+ str(dam_fun_Weitzman)  + '.png')
	# # plt.close('all')
	# plt.legend(loc='upper right')
	# plt.show()
	# # plt.clf()



# ####################################
# ## for vanilla DICE
# ## i) This part computes RS at t_tar of all deviations from the BAU path at specified time instant t_dev
# ####################################
# # fig, axes = plt.subplots(1, 1, dpi=200, figsize = (4,4))
# lince_col_dic = {1:'y', 5:'r', 9:'b', 13:'g'}
# marker_dic = {1:'o', 5:'<', 9:'>', 13:'o'}

# resol1 = 25 #single valued for grid, the higher the better for approximation
# X= np.asarray([])
# Y= np.asarray([])

# t_start = datetime.datetime.now().time()

# # miu_incr = 0.15 #constraint for incremental change of MIU
# miu_incr = 'No' #no incremental constraints
# extreme_res = {}

# #ATTENTION: Here the descending order is important! we start with the latest deviation (smallest RS)
# list_of_runs = [13, 9, 5, 1]
# for t_dev in list_of_runs:
# 	#compute and save extreme points of RS
# 	extreme_res[t_dev] = RSroutines.run_boundary(t_dev, miu_incr = miu_incr)

# 	#compute and save RS
# 	X_t, Y_t = RSroutines.RS_bulk(extreme_res[t_dev], t_dev, resol = resol1, miu_incr = miu_incr, mode = 'grid')
# 	X = np.hstack((X, X_t))
# 	Y = np.hstack((Y, Y_t))
# 	points = np.zeros([len(X), 2])

# 	for k in range(len(X)):
# 		points[k,0] = X[k]
# 		points[k,1] = Y[k]
# 		plt.plot(X[k],Y[k], color = lince_col_dic[t_dev], marker=marker_dic[t_dev], markersize=1)
# 	hull = scipy.spatial.ConvexHull(points)
# 	np.savetxt('plots_data/vanilla_DICE_plots_data/'+str(t_dev)+".csv", points[hull.vertices, ], delimiter=",")

# 	line = plt.Polygon(points[hull.vertices, ], fill=None, edgecolor= lince_col_dic[t_dev] )
# 	if miu_incr != 'No':
# 		plt.title('vanillaDICE_miu_incr run ' + str(miu_incr))
# 	plt.gca().add_patch(line)

# print(t_start, datetime.datetime.now().time())
# # plt.savefig( 'paper_plots/vanillaDICE_RS 13-9-5-1 iter = ' + str(resol1) + 'miu_incr'+ str(miu_incr)+ '.png')
# # plt.close('all')
# # # plt.show()
# # plt.clf()
