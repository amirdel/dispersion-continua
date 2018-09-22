
import numpy as np
import cPickle as cPickle
import os
import matplotlib.pyplot as plt
from py_dp.dispersion.dispersion_visualization_tools import  plot_moment_zoom, \
    plot_plume_evolution_histogram, plot_plume_x_side_y, plot_plume_x_side_y_oneTime, plot_bt, \
    plot_plume_2d_from_saved, plot_msd_com_both_one, plot_plume_2d_with_kde, plot_skew_kurt_both_one, \
    plot_plume_2d_from_saved_grid

def plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                 model_labels, l, theta, y_correction, lw, fmt, moments=True, plumes=True, bt = True, two_d=False,
                 zoom_plots=False, n_pores=500, bt_bound_box=None, delta_y = None, hists=True, trajectories=False,
                 higher_moments=True, ly=None, t_normalizer=1.0, nxbins=150, nybins=150):
    if not stencil_dt:
        stencil_dt = 1e-3
    # load target time array for plume images
    target_time_path = os.path.join(data_save_folder, 'target_times.npz')
    # if it does not exist just make the files
    if not os.path.exists(target_time_path):
        print 'Warning: target times for plumes not saved.'
        print 'trying to set reasonable times and saving the file'
        n_steps = 5
        target_time_array = np.linspace(stencil_dt, t_end, n_steps)[1:]
        target_time_array = np.floor(target_time_array / (t_normalizer * dt_mean)) * t_normalizer * dt_mean
        target_time_validation_flag = np.zeros(len(target_time_array))
        # make the msd time files
        save_path = os.path.join(data_save_folder, 'model_moments' + '.npz')
        model_moments = np.load(save_path)
        msd_x_model_array = model_moments['msd_x']
        n_points = len(msd_x_model_array[0])
        target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
        target_time_array_model = np.exp(target_time_array_model) * dt_mean
        save_path = os.path.join(data_save_folder, 'data_moments' + '.npz')
        data_moments = np.load(save_path)
        msd_x_data = data_moments['msd_x']
        n_points = len(msd_x_data) - 1
        target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
        target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean
        np.savez(target_time_path, target_time_array=target_time_array,
                 target_time_validation_flag=target_time_validation_flag,
                 target_time_array_data=target_time_array_data,
                 target_time_array_model=target_time_array_model)
        print 'target time file saved successfully!'
    target_file = np.load(target_time_path)
    target_time_array = target_file['target_time_array']
    try:
        target_time_validation_flag = target_file['target_time_validation_flag']
    except:
        target_time_validation_flag = np.zeros(len(target_time_array))
    stencil_labels = model_labels
    t_scale = dt_mean
    if ly is None:
        l_scale = l
    else:
        l_scale = ly
    # binning extents
    xmin = 0.0
    xmax = n_pores * l * np.cos(theta)
    # plot the evolution first and second moment of the plume
    if moments:
        print 'plotting moments...'
        data_name = 'model_moments'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        model_moments = np.load(save_path)
        com_x_model_array = model_moments['com_x']
        com_y_model_array = model_moments['com_y']
        msd_x_model_array = model_moments['msd_x']
        msd_y_model_array = model_moments['msd_y']
        # set the time arrays
        if higher_moments:
            skew_x_model_array = model_moments['skew_x']
            skew_y_model_array = model_moments['skew_y']
            kurt_x_model_array = model_moments['kurt_x']
            kurt_y_model_array = model_moments['kurt_y']
        data_name = 'data_moments'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        data_moments = np.load(save_path)
        com_x_data = data_moments['com_x']
        com_y_data = data_moments['com_y']
        msd_x_data = data_moments['msd_x']
        msd_y_data = data_moments['msd_y']
        if higher_moments:
            skew_x_data = data_moments['skew_x']
            skew_y_data = data_moments['skew_y']
            kurt_x_data = data_moments['kurt_x']
            kurt_y_data = data_moments['kurt_y']
        # load the time arrays for moment plots
        target_file = np.load(target_time_path)
        target_time_array_data = target_file['target_time_array_data']
        target_time_array_model = target_file['target_time_array_model']
        print 'plotting the moments of the plume...'
        axis_dict = {'ylabel1': 'longitudinal MSD', 'ylabel2': 'longitudinal COM'}
        # axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$logitudinal\;COM$"}
        save_prefix = ['COM_x_', 'MSD_x_']
        plot_msd_com_both_one(target_time_array_model, com_x_model_array, msd_x_model_array, target_time_array_data,
                              com_x_data, msd_x_data, save_folder, save_name, save_prefix, axis_dict,
                              datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

        axis_dict = {'ylabel1': r"$transverse\;MSD$", 'ylabel2': r"$transverse\;COM$", 'ymin': 400, 'ymax': 600}
        save_prefix = ['COM_y_', 'MSD_y_']
        plot_msd_com_both_one(target_time_array_model, com_y_model_array, msd_y_model_array, target_time_array_data,
                              com_y_data, msd_y_data, save_folder, save_name, save_prefix, axis_dict,
                              datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)
        if higher_moments:
            print 'plotting the 3rd and 4th moments of the plume...'
            axis_dict = {'ylabel1': r'$logitudinal\;kurtosis$', 'ylabel2': r'$logitudinal\;skewness$'}
            # axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$logitudinal\;COM$"}
            save_prefix = ['skew_x_', 'kurt_x_']
            plot_skew_kurt_both_one(target_time_array_model, skew_x_model_array, kurt_x_model_array, target_time_array_data,
                                  skew_x_data, kurt_x_data, save_folder, save_name, save_prefix, axis_dict,
                                  datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

            axis_dict = {'ylabel1': r"$transverse\;kurtosis$", 'ylabel2': r"$transverse\;skewness$"}
            save_prefix = ['skew_y_', 'kurt_y_']
            plot_skew_kurt_both_one(target_time_array_model, skew_y_model_array, kurt_y_model_array, target_time_array_data,
                                  skew_y_data, kurt_y_data, save_folder, save_name, save_prefix, axis_dict,
                                  datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

        # axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$transverse\;MSD$"}
        if zoom_plots:
            axis_dict = {'ylabel1': 'longitudinal MSD'}
            save_prefix = 'msd_x'
            # plot_moment_inset(target_time_array_data, msd_x_data, msd_y_data, datalabel, target_time_array_model,
            #               msd_x_model_array, msd_y_model_array, model_labels, t_scale, [], [],
            #               save_folder, axis_dict, lw=lw, fmt=fmt)
            print target_time_array_data/dt_mean
            zoom_box = [[100,180],[30,45]]
            zoom = 3.2
            plot_moment_zoom(target_time_array_data, msd_x_data, datalabel, target_time_array_model,
                             msd_x_model_array, model_labels, t_scale, zoom_box, zoom,
                             save_folder, save_prefix, axis_dict, lw=1, fmt=fmt, legloc=2)

            # axis_dict = {'ylabel1': r"$transverse\;MSD$"}
            axis_dict = {'ylabel1': 'transverse MSD'}
            save_prefix = 'msd_y'
            # zoom_box = [[92, 180], [9, 16]]
            zoom_box = [[100, 180], [10, 16]]
            zoom = 3.2
            plot_moment_zoom(target_time_array_data, msd_y_data, datalabel, target_time_array_model,
                             msd_y_model_array, model_labels, t_scale, zoom_box, zoom,
                             save_folder, save_prefix, axis_dict, lw=1, fmt=fmt, zoomloc=2, cor1=1, cor2=3, legloc=4)
        print 'done'

    if plumes:
        # plot the plume spreading in x and y direction
        # keep a copy of target time array
        target_time_array_all = np.copy(target_time_array)
        # separate val times from train times
        val_times = target_time_array[target_time_validation_flag>0]
        target_time_array = target_time_array[target_time_validation_flag==0]
        data_name = 'data_plumes'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            data_plumes = cPickle.load(infile)
        data_plume_x_array, data_plume_y_array = data_plumes[0], data_plumes[1]
        data_name = 'model_plumes'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            model_plumes = cPickle.load(infile)
        model_plume_x_array, model_plume_y_array = model_plumes[0], model_plumes[1]
        data_labels = [datalabel]

        print 'plotting the plume spreading in:'
        print 'x direction...'
        # plotting extents
        x_min_plot = 0.0
        x_max_plot = xmax / l_scale
        attrib = 'x'
        figsize = [6, 4]
        plot_plume_evolution_histogram(target_time_array, nxbins, xmin, xmax, attrib,
                                       save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
                                       data_labels, save_name, x_min_plot, x_max_plot, l_scale, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')
        zoom = False
        for jj in range(len(target_time_array_all)):
            if jj == len(target_time_array_all) - 1:
                if zoom_plots:
                    zoom = True
            x_maxs = xmax
            x_maxs_plot = x_max_plot
            if target_time_validation_flag[jj]:
                x_maxs *= 2
                x_maxs_plot *=2
            plot_plume_evolution_histogram(target_time_array_all, nxbins, xmin, x_maxs, attrib, save_folder,
                                           data_plume_x_array, model_plume_x_array, stencil_labels, data_labels,
                                           save_name, x_min_plot, x_maxs_plot, l_scale, t_scale=dt_mean, figsize=figsize,
                                           tidx=jj, lw=lw, fmt=fmt, zoom=zoom)

        print 'y direction...'
        attrib = 'y_array'
        # binning extents
        com_const = y_correction
        if com_const:
            delta = 0.15 * com_const
        else:
            delta = 0.12*xmax
        ymin = (com_const - delta)
        ymax = (com_const + delta)
        # plotting extents
        y_min_plot = ymin / l_scale
        y_max_plot = ymax / l_scale
        attrib = 'y'
        plot_plume_evolution_histogram(target_time_array, nybins, ymin, ymax, attrib,
                                       save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
                                       data_labels, save_name, y_min_plot, y_max_plot, l_scale, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')

        for jj in range(len(target_time_array)):
            plot_plume_evolution_histogram(target_time_array, nybins, ymin, ymax, attrib, save_folder,
                                           data_plume_y_array, model_plume_y_array, stencil_labels, data_labels,
                                           save_name, y_min_plot, y_max_plot, l_scale, t_scale=dt_mean, figsize=figsize,
                                           tidx=jj,
                                           lw=lw, fmt=fmt)
        print 'done...'

        print 'plotting side by side plumes...'
        # nxbins = nbins
        # nybins = nbins
        for i in range(len(target_time_array_all)):
            t_target = target_time_array_all[i]
            model_x_plume_list = []
            model_y_plume_list = []
            data_x_plume = data_plume_x_array[0][i, :]
            data_y_plume = data_plume_y_array[0][i, :]
            for j in range(len(model_plume_x_array)):
                model_x_plume_list.append(model_plume_x_array[j][i, :])
                model_y_plume_list.append(model_plume_y_array[j][i, :])
            x_maxs = xmax
            y_maxs = ymax
            x_maxs_plot = x_max_plot
            if target_time_validation_flag[i]:
                x_maxs *= 2
                x_maxs_plot *= 2
                y_maxs *= 1.2
            plot_plume_x_side_y_oneTime(t_target, nxbins, nybins, xmin, x_maxs, ymin, y_maxs, data_x_plume, data_y_plume,
                                        model_x_plume_list, model_y_plume_list, model_labels, t_scale, l_scale,
                                        lw, fmt, save_folder)

        plot_plume_x_side_y(target_time_array, nxbins, nybins, xmin, xmax, ymin, ymax, data_plume_x_array,
                            data_plume_y_array,
                            model_plume_x_array, model_plume_y_array, model_labels, t_scale, l_scale,
                            lw, fmt, save_folder)
        print 'done...'
        ## check for validation target times
    if bt:
        print 'plot time to get to a location'
        data_name = 'data_bt'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            data_bt_array = cPickle.load(infile)
        data_name = 'model_bt'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            model_bt_array = cPickle.load(infile)
        l_frac_file = np.load(os.path.join(data_save_folder, 'l_frac.npz'))
        l_frac_array = l_frac_file['l_frac_array']
        ## for each target length, for all models make the curve
        zoom = False
        for idx_x, target_frac in enumerate(l_frac_array):
            input_array = []
            label_array = [r"$data$"]
            input_array.append(data_bt_array[0][idx_x, :])
            for idx_model in range(len(model_labels)):
                input_array.append(model_bt_array[idx_model][idx_x, :])
                label_array.append(model_labels[idx_model])
            plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw, zoom=zoom,
                    bound_box=bt_bound_box)
            if idx_x == len(l_frac_array)-1:
                zoom = True
            if zoom_plots:
                plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw, zoom=zoom)
            # plot_bt_logscale(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
    if two_d:
        # plot the average plume in 2d
        print 'generating 2d plume figures...'
        #turn of latex rendering, causes issues on the cluster
        # plt.rc('text', usetex=False)
        data_name = 'xy_contour'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        X, Y = loader['X'], loader['Y']
        data_name = 'z_contour'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        Z, Z2 = loader['dataZ'], loader['modelZ']
        data_name = 'ycorrections'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        y_center, dy = loader['y_center'], loader['dy']
        # nlevels = 6
        nlevels = 6
        plot_plume_2d_from_saved(target_time_array, nlevels, X, Y, Z, Z2, y_center, dy, save_folder, save_name,
                                 dt_mean, scale_str=r"\overline{\delta t}$", l_scale=l_scale, fmt='png')
        plot_plume_2d_from_saved_grid(target_time_array, nlevels, X, Y, Z, Z2, y_center, dy, save_folder, save_name,
                                 dt_mean, scale_str=r"\overline{\delta t}$", l_scale=l_scale, fmt=fmt)
        print 'done'

    if hists:
        print 'plotting velocity and angle histogram...'
        save_name = 'data_hists'
        save_path = os.path.join(data_save_folder, save_name + '.npz')
        loader = np.load(save_path)
        data_hist_v, data_hist_theta = loader['hist_v'], loader['hist_theta']
        v_edges, theta_edges = loader['v_edges'], loader['theta_edges']
        mid_v = 0.5*np.diff(v_edges) + v_edges[:-1]
        mid_theta = 0.5*np.diff(theta_edges) + theta_edges[:-1]
        save_name = 'model_hists'
        save_path = os.path.join(data_save_folder, save_name + '.npz')
        loader = np.load(save_path)
        model_hist_v_array, model_hist_theta_array = loader['hist_v'], loader['hist_theta']

        fig,ax = plt.subplots(1,1)
        ax.step(mid_v, data_hist_v, where = 'mid', label='data')
        for model_hist_v, label in zip(model_hist_v_array, model_labels):
            ax.step(mid_v, model_hist_v, where = 'mid', label=label)
        ax.legend(loc='best')
        ax.set_xlabel('log(v)')
        fig.savefig(os.path.join(save_folder, 'v_hist.png'), format='png')
        plt.close(fig)
        fig, ax = plt.subplots(1, 1)
        ax.step(mid_theta, data_hist_theta, where='mid', label='data')
        for model_hist_theta, label in zip(model_hist_theta_array, model_labels):
            ax.step(mid_theta, model_hist_theta, where= 'mid', label=label)
        ax.legend(loc='best')
        ax.set_xlabel(r'$\theta$')
        fig.savefig(os.path.join(save_folder, 'theta_hist.png'), format='png')
        plt.close(fig)

    if trajectories:
        # make subfolder for trajectories
        traj_save_folder = os.path.join(data_save_folder, 'traj_data')
        print 'generating plot for comparison of v, theta trajectories btw model and averaged data'
        # save 10 trajectories
        # Assuming there is one model
        for plot_variable in ['Theta', 'V', 'DX', 'DY']:
            file_name = 'data_traj_'+plot_variable+'.npz'
            file_path = os.path.join(traj_save_folder, file_name)
            data_holder = np.load(file_path)
            target, t_array, min_length, min_val, max_val = data_holder['target'], data_holder['t_array'], \
                                                            data_holder['min_length'], data_holder['min_val'], \
                                                            data_holder['max_val']
            n_traj = len(target)
            file_name = 'model_traj_' + plot_variable + '.npz'
            file_path = os.path.join(traj_save_folder, file_name)
            data_holder = np.load(file_path)
            model_target = data_holder['target']
            fig, axes = plt.subplots(n_traj, 2, figsize=[12, n_traj * 2])
            # plot averaged data
            for i in range(n_traj):
                curr = target[i][:min_length]
                ax = axes[i][0]
                ax.plot(t_array, curr)
            title = axes[0][0].set_title(plot_variable + ' MC averaged')
            # plot the model
            for idx, traj in enumerate(model_target):
                # assuming here that the model has longer trajectories
                curr = traj[:min_length]
                ax = axes[idx][1]
                ax.plot(t_array[:len(curr)], curr)
            title = axes[0][1].set_title(plot_variable + ' stochastic model')
            for i in range(2):
                label = axes[-1][i].set_xlabel(r'$t/ \overline(\delta t)$', fontsize=14)
            for ax in axes.flatten():
                ax.set_ybound([min_val, max_val])
            figname = 'trajectory_' + plot_variable + '.png'
            fig.savefig(os.path.join(save_folder, figname), format='png')
            plt.close(fig)