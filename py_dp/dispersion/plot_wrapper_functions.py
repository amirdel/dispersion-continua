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
import cPickle as cPickle
import os
import matplotlib.pyplot as plt
from py_dp.dispersion.dispersion_visualization_tools import plume_msd_com_multiple_times, \
    plume_location_multiple_times, plume_bt_multiple_locations, save_plume_2d_with_kde

def generate_plot_data(t_end, t_scale, stencil_dt, data_save_folder, model_array, data, l, theta,
                       n_points_moments=12, n_steps_plumes=5, moments=True, plumes=True, bt=True, two_d=False,
                       kdepoints = 200000, n_pores=500, bt_bound_box=None, hists=True, avg_data=None,
                       trajectories = True, avg_data_folder=None, plume_target_times=None, plume_target_flag=None,
                       moment_data_target_times=None, moment_model_target_times=None, l_frac_array=None):
    if not stencil_dt:
        stencil_dt = 1e-3
    # save t_end, t_scale, stencil_dt. Useful for the plots.
    time_file_path = os.path.join(data_save_folder, 'time_file.npz')
    np.savez(time_file_path, t_end=t_end, t_scale=t_scale, stencil_dt=stencil_dt)
    # save l. theta
    network_spec_file = os.path.join(data_save_folder, 'network_specs.npz')
    np.savez(network_spec_file, l=l, theta=theta)
    xmax = n_pores * l * np.cos(theta)
    dt_mean = t_scale
    # set plotting times for plumes and moments
    if plume_target_times is None:
        print 'Warning: target times for plumes not specified.'
        print 'trying to set reasonable times...'
        target_time_array = np.linspace(stencil_dt, t_end, n_steps_plumes)[1:]
        target_time_array = np.floor(target_time_array / (1.0 * dt_mean)) * 1.0 * dt_mean
        # flag used to specify whether the target time is for validation
        target_time_validation_flag = np.zeros(len(target_time_array))
    else:
        target_time_array = plume_target_times
    if plume_target_flag is None:
        target_time_validation_flag = np.zeros(len(target_time_array))
    else:
        target_time_validation_flag = plume_target_flag
    n_points = n_points_moments
    print 'calculating plume moments...'
    if moment_data_target_times is None:
        print 'Warning: target times for data moments not specified.'
        print 'trying to set reasonable times...'
        target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
        target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean
    else:
        target_time_array_data = moment_data_target_times
    if moment_model_target_times is None:
        print 'Warning: target times for model moments not specified.'
        print 'trying to set reasonable times...'
        target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
        target_time_array_model = np.exp(target_time_array_model) * dt_mean
    else:
        target_time_array_model = moment_model_target_times
    # save plotting times for plumes and moments
    target_times_path = os.path.join(data_save_folder, 'target_times.npz')
    np.savez(target_times_path, target_time_array=target_time_array,
             target_time_validation_flag=target_time_validation_flag,
             target_time_array_data=target_time_array_data,
             target_time_array_model=target_time_array_model)
    # plot the evolution first and second moment of the plume
    if moments:
        com_x_model_array = []
        com_y_model_array = []
        msd_x_model_array = []
        msd_y_model_array = []
        kurt_x_model_array, kurt_y_model_array, skew_x_model_array, \
        skew_y_model_array = [[] for _ in range(4)]
        for model in model_array:
            com_x_model, msd_x_model, com_y_model, msd_y_model, skew_x_model, skew_y_model, kurt_x_model,\
            kurt_y_model = plume_msd_com_multiple_times(target_time_array_model, model)
            for parent, child in zip([com_x_model_array, com_y_model_array, msd_x_model_array,
                                      msd_y_model_array, skew_x_model_array, skew_y_model_array,
                                      kurt_x_model_array, kurt_y_model_array],
                                     [com_x_model, com_y_model, msd_x_model, msd_y_model,
                                      skew_x_model, skew_y_model, kurt_x_model, kurt_y_model]):
                parent.append(child)
            # com_x_model_array.append(com_x_model)
            # com_y_model_array.append(com_y_model)
            # msd_x_model_array.append(msd_x_model)
            # msd_y_model_array.append(msd_y_model)
        save_name = 'model_moments'
        save_path = os.path.join(data_save_folder, save_name+'.npz')
        np.savez(save_path, com_x=com_x_model_array, com_y=com_y_model_array, msd_x=msd_x_model_array,
                 msd_y=msd_y_model_array, skew_x=skew_x_model_array, skew_y=skew_y_model_array,
                 kurt_x=kurt_x_model_array, kurt_y=kurt_y_model_array)
        com_x_data, msd_x_data, com_y_data, msd_y_data, skew_x_data, skew_y_data, kurt_x_data,\
            kurt_y_data = plume_msd_com_multiple_times(target_time_array_data, data)
        save_name = 'data_moments'
        save_path = os.path.join(data_save_folder, save_name + '.npz')
        np.savez(save_path, com_x=com_x_data, com_y=com_y_data, msd_x=msd_x_data, msd_y=msd_y_data,
                 skew_x=skew_x_data, skew_y=skew_y_data, kurt_x=kurt_x_data, kurt_y=kurt_y_data)
    if plumes:
        print "calculating the plume spreading in x and y direction"
        n_steps = n_steps_plumes
        data_plume_x_array = []
        data_plume_y_array = []
        data_labels = []
        #no loop needed for data
        xtemp, ytemp = plume_location_multiple_times(target_time_array, data.x_array,
                                                     data.y_array, data.t_array)
        data_plume_x_array.append(xtemp)
        data_plume_y_array.append(ytemp)
        data_labels.append(data.label)
        #loop for model
        model_plume_x_array = []
        model_plume_y_array = []
        stencil_labels = []
        for model in model_array:
            xtemp, ytemp = plume_location_multiple_times(target_time_array, model.x_array,
                                                         model.y_array, model.t_array)
            model_plume_x_array.append(xtemp)
            model_plume_y_array.append(ytemp)
            stencil_labels.append(model.label)
        save_name = 'data_plumes'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        out = [data_plume_x_array, data_plume_y_array]
        with open(save_path, 'wb') as outfile:
            cPickle.dump(out, outfile, cPickle.HIGHEST_PROTOCOL)
        save_name = 'model_plumes'
        save_path = os.path.join(data_save_folder, save_name+'.pkl')
        out = [model_plume_x_array, model_plume_y_array]
        with open(save_path, 'wb') as outfile:
            cPickle.dump(out, outfile, cPickle.HIGHEST_PROTOCOL)
    if bt:
        print 'BT curves: calculating time to a given location'
        if l_frac_array is None:
            l_frac_array  = np.array([0.25, 0.5, 0.75])
        # save l_frac_array, multiples of one domain length used for plotting
        l_frac_path = os.path.join(data_save_folder, 'l_frac.npz')
        np.savez(l_frac_path, l_frac_array=l_frac_array)
        target_x_array = l_frac_array*xmax
        data_bt_array = []
        data_labels = []
        #no loop needed for data
        # for data in data_array:
        print 'data BT'
        ttemp = plume_bt_multiple_locations(target_x_array, data.x_array, data.t_array)
        data_bt_array.append(ttemp)
        data_labels.append(data.label)
        model_bt_array = []
        stencil_labels = []
        print 'model BT'
        for model in model_array:
            ttemp = plume_bt_multiple_locations(target_x_array, model.x_array,
                                                model.t_array)
            model_bt_array.append(ttemp)
            stencil_labels.append(model.label)
        save_name = 'data_bt'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        with open(save_path, 'wb') as outfile:
            cPickle.dump(data_bt_array, outfile, cPickle.HIGHEST_PROTOCOL)
        save_name = 'model_bt'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        with open(save_path, 'wb') as outfile:
            cPickle.dump(model_bt_array, outfile, cPickle.HIGHEST_PROTOCOL)
    if two_d:
        #plot the average plume in 2d
        print 'generating 2d plume data only for the first model...'
        plt.rc('text', usetex=False)
        n_query = 100j
        model = model_array[0]
        save_plume_2d_with_kde(target_time_array, n_query, model, data, data_save_folder, max_samples=kdepoints)

    if hists:
        print 'generating data for velocity and angle histogram...'
        if avg_data is None:
            print 'histograms need the average data to be passed in!'
        else:
            data_hist_v, data_hist_theta, v_edges, theta_edges  = get_v_theta_hist_avg_data(avg_data)
            model_hist_v_array , model_hist_theta_array = [[] for _ in range(2)]
            save_name = 'data_hists'
            save_path = os.path.join(data_save_folder, save_name + '.npz')
            np.savez(save_path, hist_v=data_hist_v, hist_theta=data_hist_theta, v_edges=v_edges,
                     theta_edges=theta_edges)
            for model in model_array:
                model_hist_v, model_hist_theta = get_v_theta_hist_model(model, v_edges, theta_edges)
                model_hist_v_array.append(model_hist_v)
                model_hist_theta_array.append(model_hist_theta)
            save_name = 'model_hists'
            save_path = os.path.join(data_save_folder, save_name + '.npz')
            np.savez(save_path, hist_v=model_hist_v_array, hist_theta=model_hist_theta_array)
    if trajectories:
        # make subfolder for trajectories
        traj_save_folder = os.path.join(data_save_folder, 'traj_data')
        if not os.path.exists(traj_save_folder):
            os.mkdir(traj_save_folder)
        if avg_data_folder is None:
            print 'path to average data is needed'
        else:
            print 'generating data for comparison of v, theta trajectories btw model and averaged data'
            # load average data files
            cartesian = np.load(os.path.join(avg_data_folder, 'avg_cartesian_0.npz'))
            polar = np.load(os.path.join(avg_data_folder, 'avg_polar_0.npz'))
            # save 10 trajectories
            n_traj = 8
            # Assuming there is one model
            model = model_array[0]
            model_dict = generate_model_dict(model)
            rand_idx = np.random.randint(0, model_dict['DX'].shape[0], n_traj)
            for plot_variable in ['Theta', 'V', 'DX', 'DY']:
                if plot_variable in ['DY', 'DX', 'Y']:
                    holder = cartesian
                else:
                    holder = polar
                target, t_array, min_length, min_val, max_val = get_traj_details_npz(holder, plot_variable, n_traj,
                                                                                     dt_mean)
                file_name = 'data_traj_'+plot_variable
                file_path = os.path.join(traj_save_folder, file_name)
                np.savez(file_path, target=target, t_array=t_array, min_length=min_length, min_val=min_val,
                         max_val=max_val)
                model_target = []
                for i in rand_idx:
                    model_target.append(model_dict[plot_variable][i,:])
                file_name = 'model_traj_' + plot_variable
                file_path = os.path.join(traj_save_folder, file_name)
                np.savez(file_path, target=model_target)

def generate_model_dict(model):
    '''
    make a small sample of particle trajectories and put them in a dictionary
    :param model:
    :return:
    '''
    sample_idx = np.random.randint(0, model.x_array.shape[0], 100)
    dx_model = np.diff(model.x_array[sample_idx, :])
    dy_model = np.diff(model.y_array[sample_idx, :])
    dt_model = np.diff(model.t_array[sample_idx, :]) + 1e-15
    VMatrix = np.sqrt(np.power(dx_model, 2) + np.power(dy_model, 2)) / dt_model
    thetaMatrix = np.arctan2(dy_model, dx_model)
    # put it all in one dictionary
    model_dict = {}
    model_dict['DX'] = dx_model
    model_dict['DY'] = dy_model
    model_dict['V'] = VMatrix
    model_dict['Theta'] = thetaMatrix
    return model_dict

def get_traj_details_npz(store_object, field, n_traj, t_scale):
    indices = np.random.randint(0, len(store_object['ptr'])-1, n_traj)
    ptr_array = store_object['ptr']
    target_array = []
    target_len_array = []
    min_val, max_val = np.inf, -np.inf
    for i in indices:
        start, end = int(ptr_array[i]), int(ptr_array[i+1])
        temp_target = store_object[field][start:end]
        target_array.append(temp_target)
        target_len_array.append(len(temp_target))
        min_val = min(min(temp_target), min_val)
        max_val = max(max(temp_target), max_val)
    min_length = min(target_len_array)
    dt = store_object['dt']
    dt_array = range(min_length)*dt/t_scale
    t_array = np.cumsum(dt_array)
    return target_array, t_array, min_length, min_val, max_val

def get_v_theta_hist_avg_data(avg_data, n_bins=100):
    hist_v, edges_v = np.histogram(np.log(avg_data['v_flat_f_added']), n_bins, density=True)
    hist_theta, edges_theta = np.histogram(avg_data['theta_flat_f_added'], n_bins, density=True)
    return hist_v, hist_theta, edges_v, edges_theta

def get_v_theta_hist_model(model, v_edges, theta_edges):
    v_mat = np.sqrt(np.diff(model.x_array) ** 2 + np.diff(model.y_array) ** 2)
    v_mat = v_mat / np.diff(model.t_array)
    model_v_flat = np.reshape(v_mat, (-1, 1))
    theta_mat = np.arctan2(np.diff(model.y_array), np.diff(model.x_array))
    model_theta_flat = np.reshape(theta_mat, (-1, 1))
    hist_v, _ = np.histogram(np.log(model_v_flat), v_edges, density=True)
    hist_theta, _ = np.histogram(model_theta_flat, theta_edges, density=True)
    return hist_v, hist_theta


# def plot_wrapper(t_end, dt_mean, stencil_dt, save_folder, save_name, model_array, data,
#                  model_labels, l, theta, y_correction, lw, fmt, moments=True, plumes=True, bt = True, two_d=False):
#     """
#     :param t_end: final pot time
#     :param dt_mean: average jump time from data
#     :param stencil_dt: dt used for the stencil
#     :param save_folder: main folder to save these plots
#     :param save_name: prefix name for saving
#     :param model_array:
#     :param data:
#     :param model_labels:
#     :param l:
#     :param theta:
#     :param y_correction:
#     :param lw:
#     :param fmt:
#     :return:
#     """
#     t_scale = dt_mean
#     l_scale = l
#     data.label = r'$data$'
#     data_array = [data]
#     # binning extents
#     xmin = 0.0
#     xmax = 500.0 * l * np.cos(theta)
#     # plot the evolution first and second moment of the plume
#     if moments:
#         n_points = 12
#         print 'calculating plume moments for ' + str(n_points) + ' times'
#         target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
#         target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean
#
#         target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
#         target_time_array_model = np.exp(target_time_array_model) * dt_mean
#
#         com_x_model_array = []
#         com_y_model_array = []
#         msd_x_model_array = []
#         msd_y_model_array = []
#         for model in model_array:
#             com_x_model, msd_x_model, com_y_model, msd_y_model = plume_msd_com_multiple_times(target_time_array_model,
#                                                                                               model)
#             com_x_model_array.append(com_x_model)
#             com_y_model_array.append(com_y_model)
#             msd_x_model_array.append(msd_x_model)
#             msd_y_model_array.append(msd_y_model)
#         com_x_data, msd_x_data, com_y_data, msd_y_data = plume_msd_com_multiple_times(target_time_array_data, data)
#
#         print 'plotting the moments of the plume...'
#         # axis_dict = {'ylabel1': 'longitudinal MSD', 'ylabel2': 'longitudinal COM'}
#         axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$logitudinal\;COM$"}
#         save_prefix = 'x_'
#         plot_msd_com_both_one(target_time_array_model, com_x_model_array, msd_x_model_array, target_time_array_data,
#                               com_x_data, msd_x_data, save_folder, save_name, save_prefix, axis_dict,
#                               data.label, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)
#
#         # axis_dict = {'ylabel1': 'transverse MSD', 'ylabel2': 'transverse COM', 'ymin': 400, 'ymax': 600}
#         axis_dict = {'ylabel1': r"$transverse\;MSD$", 'ylabel2': r"$transverse\;COM$", 'ymin': 400, 'ymax': 600}
#
#         save_prefix = 'y_'
#         plot_msd_com_both_one(target_time_array_model, com_y_model_array, msd_y_model_array, target_time_array_data,
#                               com_y_data, msd_y_data, save_folder, save_name, save_prefix, axis_dict,
#                               data.label, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)
#         print 'done'
#
#     if plumes:
#         # plot the plume spreading in x and y direction
#         n_steps = 5
#         target_time_array = np.linspace(stencil_dt, t_end, n_steps)[1:]
#         target_time_array = np.floor(target_time_array / (10.0 * dt_mean)) * 10.0 * dt_mean
#         data_plume_x_array = []
#         data_plume_y_array = []
#         data_labels = []
#         for data in data_array:
#             xtemp, ytemp = plume_location_multiple_times(target_time_array, data.x_array,
#                                                          data.y_array, data.t_array)
#             data_plume_x_array.append(xtemp)
#             data_plume_y_array.append(ytemp)
#             data_labels.append(data.label)
#         model_plume_x_array = []
#         model_plume_y_array = []
#         stencil_labels = []
#         for model in model_array:
#             xtemp, ytemp = plume_location_multiple_times(target_time_array, model.x_array,
#                                                          model.y_array, model.t_array)
#             model_plume_x_array.append(xtemp)
#             model_plume_y_array.append(ytemp)
#             stencil_labels.append(model.label)
#         del xtemp
#         del ytemp
#
#
#         print 'plotting the plume spreading in:'
#         print 'x direction...'
#         nbins = 150
#         # plotting extents
#         x_min_plot = 0.0
#         x_max_plot = xmax / l
#         attrib = 'x'
#         figsize = [6, 4]
#         plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
#                                        save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
#                                        data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, lw=lw, fmt=fmt)
#
#         plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
#                                        save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
#                                        data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean,
#                                        lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')
#
#         for jj in range(len(target_time_array)):
#             plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib, save_folder,
#                                            data_plume_x_array, model_plume_x_array, stencil_labels, data_labels,
#                                            save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, figsize=figsize, tidx=jj,
#                                            lw=lw, fmt=fmt)
#
#         print 'y direction...'
#         attrib = 'y_array'
#         # binning extents
#         com_const = y_correction
#         delta = 0.15 * com_const
#         ymin = (com_const - delta)
#         ymax = (com_const + delta)
#         nbins = 150
#         # plotting extents
#         y_min_plot = ymin / l
#         y_max_plot = ymax / l
#         attrib = 'y'
#         plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
#                                        save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
#                                        data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, lw=lw, fmt=fmt)
#
#         plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
#                                        save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
#                                        data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean,
#                                        lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')
#
#         for jj in range(len(target_time_array)):
#             plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib, save_folder,
#                                            data_plume_y_array, model_plume_y_array, stencil_labels, data_labels,
#                                            save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, figsize=figsize, tidx=jj,
#                                            lw=lw, fmt=fmt)
#         print 'done...'
#
#         print 'plotting side by side plumes...'
#         nxbins = nbins
#         nybins = nbins
#         for i in range(len(target_time_array)):
#             t_target = target_time_array[i]
#             model_x_plume_list = []
#             model_y_plume_list = []
#             data_x_plume = data_plume_x_array[0][i, :]
#             data_y_plume = data_plume_y_array[0][i, :]
#             for j in range(len(model_plume_x_array)):
#                 model_x_plume_list.append(model_plume_x_array[j][i, :])
#                 model_y_plume_list.append(model_plume_y_array[j][i, :])
#             plot_plume_x_side_y_oneTime(t_target, nxbins, nybins, xmin, xmax, ymin, ymax, data_x_plume, data_y_plume,
#                                 model_x_plume_list, model_y_plume_list, model_labels, t_scale, l_scale,
#                                 lw, fmt, save_folder)
#
#         plot_plume_x_side_y(target_time_array, nxbins, nybins, xmin, xmax, ymin, ymax, data_plume_x_array, data_plume_y_array,
#                             model_plume_x_array, model_plume_y_array, model_labels, t_scale, l_scale,
#                             lw, fmt, save_folder)
#         print 'done...'
#
#     if bt:
#         print 'plot time to get to a location'
#         l_frac_array  = np.array([0.25, 0.5, 0.75])
#         target_x_array = l_frac_array*xmax
#         data_bt_array = []
#         data_labels = []
#         for data in data_array:
#             ttemp = plume_bt_multiple_locations(target_x_array, data.x_array,
#                                                        data.t_array)
#             data_bt_array.append(ttemp)
#             data_labels.append(data.label)
#         model_bt_array = []
#         stencil_labels = []
#         for model in model_array:
#             ttemp = plume_bt_multiple_locations(target_x_array, model.x_array,
#                                                 model.t_array)
#             model_bt_array.append(ttemp)
#             stencil_labels.append(model.label)
#         ## for each target length, for all models make the curve
#         for idx_x, target_frac in enumerate(l_frac_array):
#             input_array = []
#             label_array = [r"$data$"]
#             input_array.append(data_bt_array[0][idx_x, :])
#             for idx_model in range(len(model_array)):
#                 input_array.append(model_bt_array[idx_model][idx_x, :])
#                 label_array.append(model_labels[idx_model])
#             plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
#             # plot_bt_logscale(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
#     if two_d:
#         #plot the average plume in 2d
#         print 'generating 2d plume figures...'
#         plt.rc('text', usetex=False)
#         nlevels = 6
#         n_query = 100j
#         save_folder_gif = os.path.join(save_folder, 'gif_pics')
#         if not os.path.exists(save_folder_gif):
#             os.mkdir(save_folder_gif)
#         plot_plume_2d_with_kde(target_time_array, nlevels, n_query, model, data, save_folder_gif, save_name,
#                                t_scale=dt_mean, max_samples=400000, l_scale=l)
#         print 'done'