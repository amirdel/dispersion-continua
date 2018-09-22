# Copyright 2017 Amir Hossein Delgoshaie, amirdel@stanford.edu
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
import pickle
import os
import time
import matplotlib.pyplot as plt
from py_dp.dispersion.dispersion_aux_classes import dispersionSaver
from py_dp.dispersion.dispersion_models import DispModelStencilMethod, DispModelExtendedStencil, DispModelUncorrelated
from py_dp.dispersion.transition_matrix_fcns import get_trans_matrix_single_attrib_both_methods, normalize_columns
from py_dp.dispersion.dispersion_visualization_tools import compare_trans_mat, compare_trans_mat_hist, compare_trans_mat_vtheta
from py_dp.dispersion.plot_wrapper_functions import plot_wrapper_with_saved_data, generate_plot_data
from py_dp.dispersion.dispersion_visualization_tools import model_holder, data_holder
from py_dp.dispersion.mapping_input import TemporalMapInput
from py_dp.dispersion.mapping import mapping_v_theta_repeat
from py_dp.dispersion.dispersion_aux_classes import correlatedSaver3d


# The small drift in the x direction is introduced again, find out why...

n_total = 5000
theta, n_nodes = np.pi/4, 500
t_start = time.time()
coeff_array = [10.0, 20.0, 40.0, 80.0, 160.0]
# coeff_array = [10.0, 20.0]
avg_available = True
study_folder = '/home/amirhossein/research_data/cees_plots/dt_uncorrelated'
summary_data_folder = os.path.join(study_folder, 'summary_data')
summary_pics_folder = os.path.join(study_folder, 'summary_pics')
for folder in [summary_pics_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)
t_file_path = os.path.join(study_folder, 'time_file.npz')
time_file = np.load(t_file_path)
t_end, t_scale = time_file['t_end'], time_file['t_scale']
network_length_path = os.path.join(study_folder, 'network_length.npz')
network_length_file = np.load(network_length_path)
l = network_length_file['l']

time_step = 10*t_scale
model_labels = []
for i in coeff_array:
    coeff_str = str(i).split('.')[0]
    model_labels.append(''+coeff_str)
plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Stix']})
legend_size = 13
datalabel = r"$data$"
save_name = 'all'
y_correction = 0.0
lw = 1
fmt = 'pdf'
bt_bound_box = [0, 800]
plot_wrapper_with_saved_data(t_end, t_scale, time_step, summary_data_folder, summary_pics_folder, save_name,
                             datalabel, model_labels, l, theta, y_correction, lw, fmt,
                             zoom_plots=False, n_pores=n_nodes, bt_bound_box=bt_bound_box)
print '------------------------------------------------------------------------'
t_finish = time.time()
print 'Total time: {:.2f} seconds'.format(t_finish - t_start)



