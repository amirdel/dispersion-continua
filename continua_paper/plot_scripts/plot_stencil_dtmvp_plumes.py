import os as os
import matplotlib.pyplot as plt
import numpy as np

def get_histogram(plume, bins):
    n, bins = np.histogram(plume, bins=bins, density=True)
    return n


def plot_side_byside_plumes(ax, x_vals, plume_list, label_list, style_array, color_array, lw,
                            show_legend=True, legend_loc='best'):
    loop_array = range(len(plume_list))
    for i in loop_array:
        ax.plot(x_vals, plume_list[i], label=label_list[i], color=color_array[i],
                linestyle=style_array[i], lw=lw)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=13)
    else:
        ax.set_ylabel('particle density')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlabel(r'$x/l_Y$')

# specify data and figure save directories
paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
git_data_folder = os.path.join(paper_folder, 'data', 'stencil_dtmvp_comparison')
fig_save_folder = os.path.join(paper_folder, 'plots')

plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Stix']})
legend_size = 13

fmt='pdf'

dt_coefficients = [1, 4, 10, 20]
for struct in ['exp', 'gauss']:
    # load data
    save_path = os.path.join(git_data_folder, struct + '_stencil_long_compare.npz')
    data = np.load(save_path)
    x, ly, hist_stencil, hist_dtmvp = data['x'], data['ly'], data['hist_stencil'], data['hist_dtmvp']
    legend_array, style_array, color_array = data['legend_array'], data['style_array'], data['color_array']
    lw = 1
    # plot results
    fig = plt.figure(figsize=[12,4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1,2,2)
    plot_side_byside_plumes(ax, x / ly, hist_stencil, legend_array, style_array, color_array, lw, False)
    plot_side_byside_plumes(ax2, x / ly, hist_dtmvp, legend_array, style_array, color_array, lw)
    for ax_handle in [ax, ax2]:
        ax_handle.set_xbound([0, x[-1]/ly])
    for ax_handle in [ax, ax2]:
        ax_handle.set_ybound([0, 2.5e-2])
    save_path = os.path.join(fig_save_folder, struct+'_stencil_x_last'+'.'+fmt)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)
    ############################## transverse plumes ##########################################################
    # load data
    save_path = os.path.join(git_data_folder, struct + '_stencil_trans_compare.npz')
    data = np.load(save_path)
    x, ly, hist_stencil, hist_dtmvp = data['x'], data['ly'], data['hist_stencil'], data['hist_dtmvp']
    legend_array, style_array, color_array = data['legend_array'], data['style_array'], data['color_array']
    max_y = np.max(np.array(hist_dtmvp))
    # plot the results
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_side_byside_plumes(ax, x / ly, hist_stencil, legend_array, style_array, color_array, lw, False)
    plot_side_byside_plumes(ax2, x / ly, hist_dtmvp, legend_array, style_array, color_array, lw)
    for ax_handle in [ax, ax2]:
        ax_handle.set_xbound([-15, 15])
    for ax_handle in [ax, ax2]:
        ax_handle.set_ybound([0, 1.02*max_y])
        ax.set_xlabel(r'$y/l_Y$')
    save_path = os.path.join(fig_save_folder, struct + '_stencil_y_last' + '.' + fmt)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)
    ################################## FPT curves ##########################################################
    # load data
    save_path = os.path.join(git_data_folder, struct + '_stencil_cdf_compare.npz')
    data = np.load(save_path)
    t_center, cdf_stencil, cdf_dtmvp = data['t_center'], data['cdf_stencil'], data['cdf_dtmvp']
    legend_array, style_array, color_array = data['legend_array'], data['style_array'], data['color_array']
    fig = plt.figure(figsize=[12,4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1,2,2)
    plot_side_byside_plumes(ax, t_center, cdf_stencil, legend_array, style_array, color_array, lw, False)
    plot_side_byside_plumes(ax2, t_center, cdf_dtmvp, legend_array, style_array, color_array, lw, legend_loc=4)
    for ax_handle in [ax, ax2]:
        ax_handle.set_ybound([0, 1.02])
        ax_handle.set_xlabel(r'$t/\overline{\delta t}$')
    ax.set_ylabel('cumulative distribution')
    save_path = os.path.join(fig_save_folder, struct+'_stencil_bt_cdf'+'.'+fmt)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)
