import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os as os


def side_by_side_slow_fast(stencil_model, stencil_data, lag, col_array,
                           fig_save_folder, fmt='pdf', legend_size=15, prefix='v'):
    # raise lag one matrices to the power of lag
    stencil_markov = copy(stencil_model)
    for i in range(lag-1):
        stencil_markov = np.dot(stencil_model, stencil_markov)
    if prefix.startswith('v'):
        next_str = 'next velocity class'
        label_str1 = r"$T_5^{v}(i,j)$"
        label_str2 = r"${T_1^{v}(i,j)}^5$"
    else:
        next_str = 'next angle class (radians)'
        label_str1 = r"$T_5^{\theta}(i,j)$"
        label_str2 = r"${T_1^{\theta}(i,j)}^5$"
    mat_size = stencil_model.shape[0]
    if prefix == 'theta':
        ax_bound = [-np.pi, np.pi]
        index = np.linspace(-np.pi, np.pi, num=mat_size)
    else:
        ax_bound = [0,mat_size]
        index = np.linspace(0,mat_size,num=mat_size)
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for col, ax_handle in zip(col_array, [ax, ax2]):
        # plot the stencil method plots
        ax_handle.step(index, stencil_data[:,col], where='mid', label= label_str1)
        ax_handle.step(index, stencil_markov[:, col], 'g--',where='mid', label=label_str2)
    # plot labels
    ax.set_ylabel("probability")
    ax.legend(fontsize=legend_size, loc='best')
    for ax_handle in [ax, ax2]:
        ax_handle.set_xlabel(next_str)
        ax_handle.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax_handle.set_xbound(ax_bound)
    fig_name = 'sf_side_hist_'+ str(col_array[0])+ '_' + str(col_array[1]) + '.' + fmt
    file_name = os.path.join(fig_save_folder, fig_name)
    fig.savefig(file_name, format=fmt)
    plt.close(fig)