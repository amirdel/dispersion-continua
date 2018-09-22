import os as os
import matplotlib.pyplot as plt
import numpy as np

# given an axis plot msd plots on that axis. each MSD plot will have its own input time
def moment_on_axis(ax, data_moment_array, times_arrays_data, model_moment_array, time_arrays_model,
                   coeff_array, t_scale, l_scale, label_array, color_array, style_array, moment='com_x'):
    ax.plot(times_arrays_data[0]/t_scale, data_moment_array[0][moment]/l_scale, label=label_array[0],
            color=color_array[0], linestyle=style_array[0])
    max_m = max(data_moment_array[0][moment])
    for i in range(len(coeff_array)):
        t_array = time_arrays_model[i].flatten()
        m_array = model_moment_array[i][moment].flatten()
        ax.plot(t_array/t_scale, m_array/l_scale, label=label_array[i+1], color=color_array[i+1],
                linestyle=style_array[i+1])
        max_m = max(max_m, max(m_array))
    return max_m

def get_moments_plot_data(coeff_array, root_folder, is_DTMVP=False):
    data_moment_array, model_moment_array = [], []
    target_time_data_array, target_time_model_array = [], []
    subfolder_prefix = 'dist_dt_'
    get_scale = True
    for coeff in coeff_array:
        dt_folder = 'dt_' + str(coeff)
        if is_DTMVP:
            subfolder = subfolder_prefix + str(coeff)
        else:
            subfolder = 'plots'
        data_folder = os.path.join(root_folder, dt_folder, subfolder, 'pics_data')
        if get_scale:
            tfile = np.load(os.path.join(data_folder, 'time_file.npz'))
            t_scale = tfile['t_scale']
            get_scale = False
        save_path = os.path.join(data_folder, 'data_moments' + '.npz')
        data_moments = np.load(save_path)
        data_moment_array.append(data_moments)
        save_path = os.path.join(data_folder, 'model_moments' + '.npz')
        model_moments = np.load(save_path)
        model_moment_array.append(model_moments)
        target_file = np.load(os.path.join(data_folder, 'target_times.npz'))
        target_time_data_array.append(target_file['target_time_array_data'])
        target_time_model_array.append(target_file['target_time_array_model'])
    return data_moment_array, target_time_data_array, model_moment_array, target_time_model_array, t_scale

# specify data and figure save directories
paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
git_data_folder = os.path.join(paper_folder, 'data', 'moments')
fig_save_folder = os.path.join(paper_folder, 'plots')


plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Stix']})

l_scale = 8.0
coeff_array = [1, 4, 10, 20]
style_array = ['-', '-.', ':', '--', '-.', '--', ':']
color_array = ['b', 'g', 'r', 'k', 'm', 'c', 'y']
legend_array = [r'$data$']
pre = r'$\Delta t_s = '
post = '\overline{\delta t}$'
for i in coeff_array:
    legend_array.append(pre + str(i) + post)

for struct in ['exp', 'gauss']:
    root_folder = os.path.join(git_data_folder, 'dtmvp', struct, 'sigma4')
    dtmvp_data_moment_array, dtmvp_target_time_data_array, dtmvp_model_moment_array, \
    dtmvp_target_time_model_array, t_scale = \
    get_moments_plot_data(coeff_array, root_folder, is_DTMVP=True)
    moment_names = ['msd_x', 'msd_y', 'com_x']
    label_array = ['longitudinal MSD', 'transverse MSD', 'longitudinal COM']
    # ##########################################
    # load all times and msd arrays for the stencil model
    root_folder = os.path.join(git_data_folder, 'stencil', struct, 'sigma4')
    stencil_data_moment_array, stencil_target_time_data_array, stencil_model_moment_array, \
    stencil_target_time_model_array, t_scale = \
    get_moments_plot_data(coeff_array, root_folder, is_DTMVP=False)
    # ##########################################
    # load all times and msd arrays for the uncorrelated model
    root_folder = os.path.join(git_data_folder, 'uncorrelated', struct, 'sigma4')
    prefix = 'uncorr'
    uncorr_data_moment_array, uncorr_target_time_data_array, uncorr_model_moment_array, \
    uncorr_target_time_model_array, t_scale = \
    get_moments_plot_data(coeff_array, root_folder, is_DTMVP=False)
    # ###########################################
    # sidebyside DTMVP and stencil
    for moment, label in zip(['msd_x', 'msd_y', 'com_x'],
                             label_array):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])
        max1 = moment_on_axis(ax2, dtmvp_data_moment_array, dtmvp_target_time_data_array,
                       dtmvp_model_moment_array, dtmvp_target_time_model_array,
                       coeff_array, t_scale, l_scale, legend_array, color_array, style_array, moment=moment)
        max2 = moment_on_axis(ax1, stencil_data_moment_array, stencil_target_time_data_array, stencil_model_moment_array,
                       stencil_target_time_model_array,
                       coeff_array, t_scale, l_scale, legend_array, color_array, style_array, moment=moment)
        ax1.legend(fontsize=13)
        for ax in [ax1, ax2]:
            ax.set_xlabel(r'$t/\overline{\delta t}$')
            ax.set_ybound([0, max([max1/l_scale, max2/l_scale])])
        ax1.set_ylabel(label)
        save_path = os.path.join(fig_save_folder, struct + '_' + moment + '_vs_stencil' + '.png')
        fig.savefig(save_path, format='png')
        plt.close(fig)
    # ###########################################
    # sidebyside plots for uncorr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])
    max1 = moment_on_axis(ax1, uncorr_data_moment_array, uncorr_target_time_data_array, uncorr_model_moment_array, uncorr_target_time_model_array,
                       coeff_array, t_scale, l_scale, legend_array, color_array, style_array, moment='msd_x')
    max2 = moment_on_axis(ax2, uncorr_data_moment_array, uncorr_target_time_data_array, uncorr_model_moment_array, uncorr_target_time_model_array,
                       coeff_array, t_scale, l_scale, legend_array, color_array, style_array, moment='msd_y')
    ax1.legend(fontsize=13)
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$t/\overline{\delta t}$')
    ax1.set_ylabel('longitudinal MSD')
    ax2.set_ylabel('transverse MSD')
    save_path = os.path.join(fig_save_folder, struct+'_vs_uncorr' + '.png')
    fig.savefig(save_path, format='png')
    plt.close(fig)
