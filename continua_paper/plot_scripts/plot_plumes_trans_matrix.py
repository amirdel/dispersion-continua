import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from py_dp.dispersion.plot_wrapper_functions import plot_wrapper_with_saved_data
from py_dp.dispersion.plot_wrapper_plotter import plot_wrapper_with_saved_data
from py_dp.dispersion.dispersion_aux_classes import purturb_network
from scipy.stats import gaussian_kde
from py_dp.dispersion.dispersion_visualization_tools import side_by_side_thetamat_plot, theta_mat_moment_plots
from py_dp.dispersion.transition_matrix_fcns import get_trans_matrix_single_attrib, normalize_columns
from py_dp.dispersion.dispersion_visualization_tools import compare_trans_mat, compare_trans_mat_hist,\
    compare_trans_mat_vtheta, single_trans_mat, side_by_side_thetamat_hists
from py_dp.visualization.plot_sidebyside_fast_slow import side_by_side_slow_fast


plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Stix']})
legend_size = 13

# specify data and figure save directories
paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
git_data_folder = os.path.join(paper_folder, 'data', 'dtmvp_dt10')
fig_save_main = os.path.join(paper_folder, 'plots')
#
# git_data_folder = '/home/amirhossein/research_data/paper_cees/dt10_for_git'
# fig_save_main = '/home/amirhossein/Desktop/git_test'


plume_fmt = 'pdf'
mat_fmt = 'png'
###################################################################################################
# root_folder = '/home/amirhossein/research_data/paper_cees/dist_KDE_plots/'
nxbins = 100
nybins = 90
n_cells = 1024
multiplier = 10.0
l = 1.0
theta = 0.0
ly = 8.0
sigma_str = '4'
dt_string = str(multiplier).split('.')[0]
# moments = True
# plumes = True
# bt = True
# two_d = True
# traj=False
# hists = True
# tmat = True

moments = False
plumes = True
bt = False
two_d = True
traj = False
hists = False

model_labels = []
# model_str = r'$v,pdf(\theta|y) \mbox{-}' + dt_string + '\overline{\delta t}$'
model_str = r'$DTMVP-' + dt_string + '\overline{\delta t}$'
model_labels.append(model_str)
plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Stix']})
datalabel = r"$MC$"
for struct in ['exp', 'gauss']:
    data_save_folder = os.path.join(git_data_folder, struct, 'sigma4', 'dt_10', 'dist_dt_10', 'pics_data')
    savefig_path = os.path.join(fig_save_main, struct+'_plume_matrix_pics')
    if not os.path.exists(savefig_path): os.mkdir(savefig_path)
    t_file_path = os.path.join(data_save_folder, 'time_file.npz')
    time_file = np.load(t_file_path)
    t_end, t_scale = time_file['t_end'], time_file['t_scale']
    stencil_dt = multiplier * t_scale
    save_name = struct + sigma_str
    y_correction = 0.0
    lw = 1
    fmt = plume_fmt
    plot_wrapper_with_saved_data(t_end, t_scale, stencil_dt, data_save_folder, savefig_path, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt, two_d=two_d,
                                 zoom_plots=False, n_pores=n_cells, trajectories=traj, hists=hists, ly=ly,
                                 nxbins=nxbins, nybins=nybins)
    # theta mat plots:
    print 'plotting theta tmat'
    matfile = np.load(os.path.join(data_save_folder, 'matrix_plots.npz'))
    side_by_side_thetamat_plot(matfile['t_mat'], matfile['t_hist'], savefig_path,
                               matfile['mid_y'], fmt=fmt)
    # plot moments and fits
    theta_mat_moment_plots(matfile['lsq_moments'], matfile['fitted_moments'],
                           matfile['mid_y'], matfile['col_array'], ly, savefig_path, fmt=fmt)
    # plot column-wise comparison of theta mat
    col_array = [16, 30]
    side_by_side_thetamat_hists(matfile['t_hist'], matfile['t_mat'], col_array, savefig_path,
                                matfile['mid_y'], fmt=fmt, l_scale=8.0, fs=18, legend_fs=14)
    # vmat plots:
    print 'plotting V tmat'
    case_base = os.path.join(git_data_folder, struct, 'sigma4', 'dt_10', 'dist_dt_10')
    # create and save the transition matrices for v and theta for lag one and five
    matrix_data_folder = os.path.join(case_base, 'matrix_data')
    matrix_plot_folder = os.path.join(savefig_path, 'matrix_plots')
    for folder in [matrix_plot_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    lag_array = [1, 5]
    print 'extracting matrices'
    v_file = os.path.join(matrix_data_folder, 'v_list_no_freq.npy')
    v_mat_list_nofreq = np.load(v_file)
    theta_file = os.path.join(matrix_data_folder, 'theta_list_no_freq.npy')
    theta_mat_list_nofreq = np.load(theta_file)

    lag = lag_array[1]
    case_name = ''
    # plot results
    print 'plotting matrices'
    for v_list, theta_list, mat_fig_folder in zip([v_mat_list_nofreq],
                                                  [theta_mat_list_nofreq],
                                                  [matrix_plot_folder]):
        column_folder = os.path.join(mat_fig_folder, 'columnwise_CK')
        if not os.path.exists(column_folder):
            os.mkdir(column_folder)
        trans_matrix_v1 = normalize_columns(v_list[0])
        trans_matrix_v2 = normalize_columns(v_list[1])
        trans_matrix_t1 = normalize_columns(theta_list[0])
        trans_matrix_t2 = normalize_columns(theta_list[1])
        # column-wise comparison of the aggregate transition probabilities
        v_str = 'v'
        theta_str = 'theta'
        both_str = 'both_matrix'
        compare_trans_mat_hist(trans_matrix_v1, trans_matrix_v2, lag, column_folder, v_str, legend_size=14,
                               fmt=mat_fmt)
        compare_trans_mat_hist(trans_matrix_t1, trans_matrix_t2, lag, column_folder, theta_str, legend_size=14,
                               fmt=mat_fmt)
        fontsize = 14
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('xtick', labelsize=fontsize * 0.8)
        plt.rc('ytick', labelsize=fontsize * 0.8)
        compare_trans_mat(trans_matrix_v1, trans_matrix_v2, lag, mat_fig_folder, v_str, fmt=mat_fmt)
        compare_trans_mat(trans_matrix_t1, trans_matrix_t2, lag, mat_fig_folder, theta_str, fmt=mat_fmt)
        compare_trans_mat_vtheta(trans_matrix_v1, trans_matrix_t1, mat_fig_folder, both_str, fmt=mat_fmt)
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        single_trans_mat(trans_matrix_v1, mat_fig_folder, 'v_single', fmt=mat_fmt)
        plt.rcParams.update({'font.size': 20})
        side_by_side_slow_fast(trans_matrix_v1, trans_matrix_v2, lag, [4, 62],
                               mat_fig_folder, fmt=mat_fmt, legend_size=15)
##################################################################################################
# if copy_plots:
#     print 'copy all plots to folder...'
#     os.system('python copy_paper_plots_continua_png.py')
# if run_scripts:
#     os.system('python average_trajectories.py')
#     os.system('python perm_periodicity.py')
#     os.system('python MC_setup.py')

