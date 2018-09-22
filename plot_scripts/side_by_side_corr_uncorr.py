import os as os
import cPickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from py_dp.dispersion.binning import get_cdf_from_bins

def get_histogram(plume, bins):
    n, bins = np.histogram(plume, bins=bins, density=True)
    return n


def plot_side_byside_plumes(ax, x_vals, plume_list, label_list, style_array, color_array, lw,
                            show_legend=True, legend_loc='best'):
    for i in range(len(plume_list)):
        ax.plot(x_vals, plume_list[i], label=label_list[i], color=color_array[i],
                linestyle=style_array[i], lw=lw)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=13)
    else:
        ax.set_ylabel('Particle density')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlabel(r'$x/l$')


plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
# plt.rc('font',**{'family':'serif','serif':['Caladea']})
plt.rc('font',**{'family':'serif','serif':['Stix']})
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
legend_size = 13
fig_save_folder = '/home/amirhossein/research_data/cees_plots/paper_side_large'
fmt='pdf'


dt_coefficients = [20, 40, 80, 160]
# dt_coefficients = [10]
data_folders = []
folder_pre = '/home/amirhossein/research_data/cees_plots/dt_study_mean/dt_'
folder_post = '/plots/00_none/plumes_data'
for coeff in dt_coefficients:
    data_folders.append(folder_pre+str(coeff)+folder_post)
# data_folder_5 = '/home/amirhossein/research_data/cees_plots/dt_study_mean/dt_5/plots/00_none/plumes_data'
# data_folder_10 = '/home/amirhossein/research_data/cees_plots/dt_study_mean/dt_10/plots/00_none/plumes_data'
# data_folder_2 = '/home/amirhossein/research_data/cees_plots/dt_study_mean/dt_2/plots/00_none/plumes_data'

# load l
network_length_path = os.path.join(data_folders[0], 'network_specs.npz')
network_length_file = np.load(network_length_path)
l = network_length_file['l']
# load time scale
t_file_path = os.path.join(data_folders[0], 'time_file.npz')
time_file = np.load(t_file_path)
t_end, t_scale = time_file['t_end'], time_file['t_scale']
# load the plumes files
plumes_data_path = os.path.join(data_folders[0], 'data_plumes.pkl')
with open(plumes_data_path, 'rb') as infile:
    data_plumes = cPickle.load(infile)
plumes = []
# first entry is the data plume
plumes.append(data_plumes)
for folder in data_folders:
    save_path = os.path.join(folder, 'model_plumes' + '.pkl')
    with open(save_path, 'rb') as infile:
        model_plume = cPickle.load(infile)
    plumes.append(model_plume)

# print len(plumes)

xmin = 0
xmax = 500*l*np.cos(np.pi/4)
nbins = 150
bins = np.linspace(xmin,xmax,nbins)
x = bins[:-1] + 0.5*np.diff(bins)
plot_idx = -1
# pack data, 2d_10, 2d_5
hist_stencil = []
inds = np.zeros(len(dt_coefficients)+1, dtype=np.int)
for i,array in zip(inds, plumes):
    hist_stencil.append(get_histogram(array[0][i][plot_idx], bins))

inds = np.ones(len(dt_coefficients)+1, dtype=np.int)
inds[0] = 0
# pack data, 3d_10, 3d_20
hist_extended = []
for i,array in zip(inds, plumes):
    hist_extended.append(get_histogram(array[0][i][plot_idx], bins))

## load the un-correlated case
uncorr_folder = '/home/amirhossein/research_data/cees_plots/dt_uncorrelated/summary_data'
save_path = os.path.join(uncorr_folder, 'model_plumes' + '.pkl')
with open(save_path, 'rb') as infile:
    uncorr_plume = cPickle.load(infile)
hist_uncorr = []
# add the data plume
hist_uncorr.append(get_histogram(data_plumes[0][0][plot_idx], bins))
# add all the other plumes
for i in range(len(dt_coefficients)):
    hist_uncorr.append(get_histogram(uncorr_plume[0][i][plot_idx], bins))

style_array = ['-', '-.', '--', ':', '-.', '--', ':']
color_array = ['b', 'g', 'r', 'k', 'm', 'c', 'y']
lw = 1
# legend_array = [r'$data$', r'$\Delta t_s = 10 \overline{\delta t}$', r'$\Delta t_s = 5 \overline{\delta t}$',
#                 r'$\Delta t_s = 2 \overline{\delta t}$']
legend_array = [r'$data$']
pre = r'$\Delta t_s = '
post = '\overline{\delta t}$'
for i in dt_coefficients:
    legend_array.append(pre + str(i) + post)

fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1,2,2)
# print len(legend_array)
# plot_side_byside_plumes(ax, x/l, hist_stencil, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax, x/l, hist_uncorr, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax2, x/l, hist_extended, legend_array, style_array, color_array, lw)
# plt.show()
for ax_handle in [ax, ax2]:
    ax_handle.set_ybound([0, 1.2e-2])
save_path = os.path.join(fig_save_folder, 'ind_spread_x_all_oneTime_320'+'.'+fmt)
fig.savefig(save_path, format=fmt)
plt.close(fig)

fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1,2,2)
# print len(legend_array)
# plot_side_byside_plumes(ax, x/l, hist_stencil, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax, x/l, hist_stencil, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax2, x/l, hist_extended, legend_array, style_array, color_array, lw)
# plt.show()
save_path = os.path.join(fig_save_folder, 'plumes_side_large_dt'+'.'+fmt)
fig.savefig(save_path, format=fmt)
plt.close(fig)


### FPT curves
# data_name = 'data_bt'
# save_path = os.path.join(data_folder_5, data_name + '.pkl')
# with open(save_path, 'rb') as infile:
#     data_bt = cPickle.load(infile)
# data_name = 'model_bt'
# save_path = os.path.join(data_folder_5, data_name + '.pkl')
# with open(save_path, 'rb') as infile:
#     model_bt_5 = cPickle.load(infile)
# save_path = os.path.join(data_folder_10, data_name + '.pkl')
# with open(save_path, 'rb') as infile:
#     model_bt_10 = cPickle.load(infile)

# bt_arrays = [data_bt, model_bt_10, model_bt_5, model_bt_2]
# file_names = ['data_bt', 'model_bt', 'model_bt', 'model_bt']
# result_folders = [data_folder_2, data_folder_10, data_folder_5, data_folder_2]
result_folders = [data_folders[0]]
for item in data_folders:
    result_folders.append(item)
file_names = ['model_bt' for i in range(len(data_folders)+1)]
file_names[0] = 'data_bt'
bt_arrays = [[] for i in range(len(data_folders)+1)]

for idx, (address, file_name) in enumerate(zip(result_folders, file_names)):
    save_path = os.path.join(address, file_name + '.pkl')
    with open(save_path, 'rb') as infile:
        bt_arrays[idx] = cPickle.load(infile)

times_stencil = np.array([])
# for i,array in zip([0,0,0], [data_bt, model_bt_10, model_bt_5]):
for i, array in zip([0, 0, 0, 0], bt_arrays):
    times_stencil = np.hstack((times_stencil, array[i][plot_idx,:]))
tmin, tmax = np.amin(times_stencil), 0.75*np.percentile(times_stencil, 99.8)
t_edges = np.linspace(tmin, tmax, 150)
t_edges/=t_scale
t_center = t_edges[:-1] + 0.5*np.diff(t_edges)
cdf_stencil = []
# for i,array in zip([0,0,0], [data_bt, model_bt_10, model_bt_5]):
idx = np.zeros(len(bt_arrays), dtype=np.int)
for i,array in zip(idx, bt_arrays):
    t_array = array[i][plot_idx,:]
    _, cdf = get_cdf_from_bins(t_array[t_array>0]/t_scale, t_edges)
    cdf_stencil.append(cdf)

cdf_extended = []
idx = np.ones(len(bt_arrays), dtype=np.int)
idx[0] = 0
# for i,array in zip([0,1,1], [data_bt, model_bt_10, model_bt_5]):
for i,array in zip(idx, bt_arrays):
    t_array = array[i][plot_idx,:]
    _, cdf = get_cdf_from_bins(t_array[t_array>0]/t_scale, t_edges)
    cdf_extended.append(cdf)


# get the CDF for the uncorrelated case
time_file = os.path.join(uncorr_folder, 'model_bt.pkl')
with open(time_file) as infile:
    uncorr_times = cPickle.load(infile)
cdf_uncorr = [cdf_stencil[0]]
for i in range(len(dt_coefficients)):
    t_array = uncorr_times[i][plot_idx,:]
    _, cdf = get_cdf_from_bins(t_array[t_array > 0] / t_scale, t_edges)
    cdf_uncorr.append(cdf)


fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1,2,2)
plot_side_byside_plumes(ax, t_center, cdf_stencil, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax2, t_center, cdf_extended, legend_array, style_array, color_array, lw, legend_loc=4)
for ax_handle in [ax, ax2]:
    ax_handle.set_ybound([0, 1.02])
    ax_handle.set_xlabel('nondimensional FPT')
ax.set_ylabel('cumulative distribution')
# plt.show()
save_path = os.path.join(fig_save_folder, 'cdf_side_large_dt'+'.'+fmt)
fig.savefig(save_path, format=fmt)
plt.close(fig)

fig = plt.figure(figsize=[12,4])
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1,2,2)
plot_side_byside_plumes(ax, t_center, cdf_uncorr, legend_array, style_array, color_array, lw, False)
plot_side_byside_plumes(ax2, t_center, cdf_extended, legend_array, style_array, color_array, lw, legend_loc=4)
for ax_handle in [ax, ax2]:
    ax_handle.set_ybound([0, 1.02])
    ax_handle.set_xlabel('nondimensional FPT')
ax.set_ylabel('cumulative distribution')
# plt.show()
save_path = os.path.join(fig_save_folder, 'ind_bt_linear_75'+'.'+fmt)
fig.savefig(save_path, format=fmt)
plt.close(fig)