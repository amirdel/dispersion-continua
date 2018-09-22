# Copyright 2018 Amir Hossein Delgoshaie, amirdel@stanford.edu
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
# is hereby granted, provided that the above copyright notice and this permission notice appear in all
# copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import numpy as np
import time
import os
import pickle
from py_dp.dispersion.realizations.multiple_realization_functions import multiple_continuum_realizations
from py_dp.dispersion.dispersion_continua import find_compatible_n_particles
from py_dp.dispersion.dispersion_visualization_tools import save_big_data_array
from py_dp.dispersion.average_trajectories import average_all_realizations
from py_dp.simulation.grid_structured import structuredGrid

# This script contains the workflow for generating a grid
# solving a flow problem and performing particle tracking
# building a DTMVP model from from the particle tracking results
# performing simulation using the DTMVP model
# comparing the DTMVP results with the MC data

t_start = time.time()
# specify a case_folder for saving the results of this script
# make sure this folder already exists on your machine
case_folder = '/home/amirhossein/Desktop/test_script'
# create a folder for saving the grid data structure
grid_save_folder = os.path.join(case_folder, 'grid_folder')
# create a folder for saving the realizations
realz_folder = os.path.join(case_folder, 'realizations')
# set path for saving model results
result_dir = os.path.join(case_folder, 'model_results')
# path for saving plume figures
savefig_path = os.path.join(case_folder, 'test_figures')
# path for saving plot data
fig_data_path = os.path.join(case_folder, 'test_figures_data')
for folder in [realz_folder, result_dir, savefig_path,
               fig_data_path, grid_save_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)
#######################################################################################
# 1. Generate and save grid
print '1. Generating grid...'
grid_size = 100
#define dx and dy (only supported if they are equal)
dx = dy = 1.0
#define number of grid cells in the x direction (m) and in the y direction (n)
m = 100
n = 100
#specify boundary type -> full-periodic or non-periodic
boundaryType = 'full-periodic'
#save file name
grid_name = str(m) + '_'+ str(n)+ '_' + boundaryType +'.pkl'
#create the grid
grid = structuredGrid(m, n, dx, dy, boundaryType=boundaryType)
#save the grid
grid_save_path = os.path.join(grid_save_folder, grid_name)
with open(grid_save_path,'wb') as output:
    pickle.dump(grid, output, pickle.HIGHEST_PROTOCOL)
print 'Grid saved successfully.'
#######################################################################################
# 2. Generate particle tracking data
print '2. Particle tracking in different realizations'
# follow each particle until it changes cells n_teps times
n_steps = 300
# path to csv file containing log-conductivity
# we assume log-conductivity has mean zero and variance one
paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
perm_path = os.path.join(paper_folder, 'continua_paper', 'data',
                         'perm_field_sample', 'exp_s100_8.csv')
# number of perm_realization in perm path
n_runs = 50
# dp/dx, dp/dy for periodic pressure solver
dp_x = 1.0
dp_y = 0.0
# combine each n_combine runs into one file
n_combine = 10
# combine each n_avg_combine files into one averaged file
n_avg_combine = 1
# number of buffer correlation length
l_corr_buffer = 2
sigma = 1.0
ly = 8
struct = 'exp'
std = sigma**0.5
split = str(sigma).split('.')
coeff = 2.0
# number of particles is pre calculated to work with the separation distance
n_particles = find_compatible_n_particles(grid_size, ly, l_corr_buffer)

#make the realizations
multiple_continuum_realizations(grid_save_path, realz_folder, perm_path,
                                dp_x, dp_y, n_particles, n_steps, n_runs, n_combine,
                                n_buffer_ly=l_corr_buffer, ly=ly, std=std)
n_mc_avg_files = int(n_runs / n_combine)
## post processing
## save the big_data_array and time scale file
save_big_data_array(realz_folder, 0, n_mc_avg_files)
big_t_path = os.path.join(realz_folder, 'big_t.npy')
big_t = np.load(big_t_path)
# find the last time where we have data on all particles
t_end = min(big_t[:, -1])
# find the mean dt for the MC data
dt_mean = np.mean(np.diff(big_t, axis=1))
assert (t_end > 0)
assert (dt_mean > 0)
assert (dt_mean < t_end)
# dump into files
time_file = os.path.join(realz_folder, 'time_file.npz')
np.savez(time_file, t_scale=dt_mean, t_end=t_end)
# save the link length
with open(os.path.join(realz_folder, 'real_0.pkl')) as input:
    data0 = pickle.load(input)
    l = (data0.x_array[0, 1] - data0.x_array[0, 0])
l_file = os.path.join(realz_folder, 'network_length.npz')
np.savez(l_file, l=l)


# load time scale and simulation end time
t_file_path = os.path.join(realz_folder, 'time_file.npz')
time_file = np.load(t_file_path)
t_end, dt_mean = time_file['t_end'], time_file['t_scale']
t_file_u_path = os.path.join(realz_folder, 't_scale_u.npz')
t_file_u = np.load(t_file_u_path)
t_scale = t_file_u['t_scale_u']
# save averaged realiztions
coeff_str = str(coeff).split('.')[0]
save_folder = os.path.join(realz_folder, 'time_averaged_' + coeff_str + 'dt')
if os.path.exists(save_folder):
    print 'warning: the average data folder already existed. re-writing the old files...'
else:
    os.mkdir(save_folder)
time_step = coeff * t_scale
average_all_realizations(realz_folder, n_mc_avg_files, time_step, save_folder,
                         n_combine=n_avg_combine, prefix='real', verbose=True,
                         print_every=5)
print '---------------------------------------------------------------'
#######################################################################################
# 3. Extract model
print '3. Extracting transition matrices and distributions for DTMVP'
from py_dp.dispersion.mapping_input import TemporalMapInputWithY
from py_dp.dispersion.mapping import mapping_v_theta_y
from py_dp.dispersion.independent_transition_models_theta_dist import TransInfoIndependentVThetaDist
from py_dp.dispersion.independent_dispersion_models_theta_dist import DispModelIndependentVThetaDist
from py_dp.dispersion.dispersion_aux_classes import dispersionSaver
import matplotlib.pyplot as plt

experiment_folder = os.path.join(case_folder, 'DTMVP')
if not os.path.exists(experiment_folder): os.mkdir(experiment_folder)
# binning options
n_absv_class = 100
n_theta_class = 100
n_y_class = 61
max_allowed = 'auto'
n_slow_class = 10
n_dtmvp_particles = 1000
avg_available = True
avg_folder = os.path.join(realz_folder, 'time_averaged_' + coeff_str + 'dt')
time_step = t_scale * coeff
dt_string = coeff_str
dt_folder = os.path.join(experiment_folder, 'dt_' + dt_string)
if not os.path.exists(dt_folder):
    os.mkdir(dt_folder)
n_steps_dtmvp = 100
# # generate the transition model
map_input = TemporalMapInputWithY(avg_folder, n_mc_avg_files, n_absv_class, n_theta_class, n_y_class,
                                  time_step, n_slow_class, max_allowed, average_available=avg_available)
# mapping contains all the functions for finding class indices
mapping = mapping_v_theta_y(map_input.v_log_edges, map_input.theta_edges, map_input.y_edges)

trans_info = TransInfoIndependentVThetaDist(avg_folder, n_mc_avg_files, mapping, map_input)
trans_mat_v, trans_mat_theta = trans_info.get_trans_matrix(1)
print '---------------------------------------------------------------'

print '4. Simulating plumes using DTMVP model'
# run model and save results
stencil_name = 'dt_' + dt_string
dt = trans_info.time_step
# mapping = trans_info.mapping
init_v_class_count, init_theta_class_count = trans_info.init_v_class_count, trans_info.init_theta_class_count
model_1 = DispModelIndependentVThetaDist(n_dtmvp_particles, n_steps_dtmvp, dt, None, trans_mat_v,
                                         trans_mat_theta, mapping, init_v_class_count,
                                         init_theta_class_count, lsq_frac=0.4)
model_1.follow_all_particles_vector(warn=False)
# dump results
result_holder = dispersionSaver(model_1.x_array, model_1.time_array, model_1.last_index_array,
                             y_array = model_1.y_array)
output_save_name = stencil_name.split('.')[0]+'.pkl'
save_address = os.path.join(result_dir, output_save_name)
with open(save_address, 'wb') as output:
    pickle.dump(result_holder, output, pickle.HIGHEST_PROTOCOL)
print '---------------------------------------------------------------'
#######################################################################################
from py_dp.dispersion.plot_wrapper_functions import generate_plot_data
from py_dp.dispersion.plot_wrapper_plotter import plot_wrapper_with_saved_data
from py_dp.dispersion.dispersion_visualization_tools import model_holder, assemble_averaged_data, data_holder
print '5. Plotting results'
data = data_holder(realz_folder)
#make sure all data streamlines start from the origin
data.y_array = data.y_array - data.y_array[:,0][:, None]
data.x_array = data.x_array - data.x_array[:,0][:, None]

stencil_names = []
stencil_names.append(stencil_name)
model_labels = []
model_str = r'$DTMVP \mbox{-}'+ dt_string+'\overline{\delta t}$'
model_labels.append(model_str)

l = 1.0
theta = 0.0
model = model_holder(os.path.join(result_dir, stencil_name+'.pkl'), model_labels[0])
model_array = [model]
# load the time averaged data
data_avg = assemble_averaged_data(avg_folder, n_mc_avg_files)
# save data for the matrix
mid_y = np.diff(mapping.y_edges) / 2 + mapping.y_edges[:-1]
np.savez(os.path.join(fig_data_path, 'matrix_plots.npz'), t_mat=model_1.theta_mat_model,
         coeff=model_1.dd, t_hist=trans_mat_theta, mid_y= mid_y, col_array= model_1.col_array,
         lsq_moments=model_1.lsq_moments, fitted_moments=model_1.fitted_moments)

generate_plot_data(t_end, t_scale, time_step, fig_data_path, model_array, data, l, theta,
                   n_pores=3*m, avg_data=data_avg, avg_data_folder=avg_folder, moments=False, bt=False)
# In[15]:
plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Stix']})
legend_size = 13
datalabel = r"$data$"
save_name = 'test'
y_correction = 0.0
lw = 1
fmt = 'png'
plot_wrapper_with_saved_data(t_end, t_scale, time_step, fig_data_path, savefig_path, save_name, datalabel,
                     model_labels, l, theta, y_correction, lw, fmt, n_pores=3*m, ly=ly, moments=False, bt=False)
print '---------------------------------------------------------------'
#######################################################################################
t_finish = time.time()
print 'figures have been saved in your_case_folder/test_figures directory'
print 'Total script time: {:.2f} seconds'.format(t_finish-t_start)
