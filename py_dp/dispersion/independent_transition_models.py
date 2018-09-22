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
from scipy.sparse import csc_matrix
from py_dp.dispersion.binning import abs_vel_log_bins_low_high, make_theta_bins_linear, make_y_bins_linear
from py_dp.dispersion.binning import make_input_for_binning_v_theta_freq, binning_input_v_theta_freq_y
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_with_freq_cython
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_cython
from py_dp.dispersion.mapping import mapping_v_theta_repeat, mapping_v_theta_y
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy
from py_dp.dispersion.average_trajectories import add_repetitions


class TransInfoIndependentSimple(object):
    """
    a class for extracting binned trasition information for 2d spatial cases with independent processes
    for the velocity p(v2|v1) and angle p(theta_2|theta_1) in time.
    """
    def __init__(self, input_folder, n_total_realz, mapping, map_input, average_available, time_step=None):
        """

        :param input_folder: the folder containing the particle tracking data
        :param n_binning_realz: number of realizations used for binning the data
        :param n_total_realz: total number of realizations
        :param n_absv_class: number of velocity classes
        :param n_theta_class: number of angle classes
        :param time_step: stencil time
        :param n_slow_class: number of slow classes (refining the velocity bins for very slow velocity values)
        :param max_allowed: max bins size allowed for log(v)
        """
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        # making sample data for creating bins
        self.mapping = mapping
        self.map_input = map_input
        self.average_available = average_available
        self.init_v_class_count, self.init_theta_class_count = self.get_init_class_count()

    def get_init_class_count(self):
        """
        :return:
         init_v_class_count: initial count of the velocity class. size (n_velocity_class,)
         init_v_theta_count: initial count of the angle class. size (n_theta_class,)
        """
        map_input = self.map_input
        mapping = self.mapping
        new_v, new_theta, new_f = remove_duplicate_xy(map_input.initial_v, map_input.initial_theta,
                                                      map_input.initial_f)
        init_v_idx = self.mapping.find_1d_class_idx(np.log(new_v), mapping.v_log_edges)
        init_theta_idx = self.mapping.find_1d_class_idx(map_input.initial_theta, map_input.theta_edges)
        init_v_class_count, init_theta_class_count = np.zeros(self.mapping.n_abs_v_classes), np.zeros(self.mapping.n_theta_classes)
        for v_idx, theta_idx in zip(init_v_idx, init_theta_idx):
            init_v_class_count[v_idx] += 1
            init_theta_class_count[theta_idx] += 1
        return init_v_class_count, init_theta_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True, avg_available=None):
        if avg_available is None:
            avg_available = self.average_available
        if avg_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else:
            return self.get_trans_matrix_from_scratch(lag)

    def get_trans_matrix_from_average(self, lag, print_every=50, verbose=True):
        dt = self.time_step
        # get the size of the transition matrices
        n_v_class, n_theta_y_class = self.mapping.n_abs_v_classes, self.mapping.n_2d_theta_y_classes
        # initialize the sparse transition matrices
        print 'getting transition matrix from averaged data...'
        i_list_v, j_list_v, val_list_v = [[] for _ in range(3)]
        i_list_theta, j_list_theta, val_list_theta = [[] for _ in range(3)]
        ij_set_v, ij_set_theta = [set([]) for _ in range(2)]
        for j in range(self.n_total_realz):
            start_idx = 0
            # load the polar coordinates file
            data_path = os.path.join(self.input_folder, 'avg_polar_' + str(j) + '.npz')
            data = np.load(data_path)
            big_v, big_theta, big_f, ptr_list = data['V'], data['Theta'], data['F'], data['ptr']
            for i in ptr_list:
                new_v, new_theta, new_f = big_v[start_idx:i], big_theta[start_idx:i], big_f[start_idx:i]
                start_idx = i
                if len(new_v)>lag:
                    # simple process for v: v1, v2, v3, ...
                    v_process_idx = self.mapping.find_1d_class_idx(np.log(new_v), self.mapping.v_log_edges)
                    # fill the transition matrix for this velocity series
                    fill_one_trajectory_sparse_with_freq_cython(lag, v_process_idx, new_f, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # fill the transition matrix for this angle series
                    theta_process_idx = self.mapping.find_1d_class_idx(new_theta, self.mapping.theta_edges)
                    fill_one_trajectory_sparse_cython(lag, theta_process_idx, i_list_theta,
                                                      j_list_theta, ij_set_theta, val_list_theta)
        print 'done.'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_v_class, n_v_class)), \
               csc_matrix((val_list_theta, (i_list_theta, j_list_theta)), shape = (n_theta_y_class, n_theta_y_class))

    def get_trans_matrix_from_scratch(self, lag):
        """
        extract the transition matrix for velocity and angle for the given lag
        :param lag: the lag used to derive the transition matrix
        :return: v_trans_matrix, theta_trans_matrix for the given lag
        """
        # get the size of the transition matrices
        n_v_class, n_theta_class = self.mapping.n_abs_v_classes, self.mapping.n_theta_classes
        # initialize the sparse transition matrices
        i_list_v, j_list_v, val_list_v = [[] for _ in range(3)]
        i_list_theta, j_list_theta, val_list_theta = [[] for _ in range(3)]
        ij_set_v, ij_set_theta = [set([]) for _ in range(2)]
        time_step = self.time_step
        print 'extracting trans matrix for the velocity and angle process...'
        for j in range(self.n_total_realz):
            if not j%5:
                print 'reading realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array) + 1e-15
            if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
                print 'some array was empty, skipping this file...'
                continue
            lastIdx = dataHolder.last_idx_array
            vxMatrix = np.divide(dx,dt)
            vyMatrix = np.divide(dy,dt)
            m = dx.shape[0]
            for i in range(m):
                x_start = dataHolder.x_array[i, 0]
                y_start = dataHolder.y_array[i, 0]
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start, time_step)
                v_temp = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))/time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                if len(v_temp) > lag:
                    new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
                    v_process_idx = self.mapping.find_1d_class_idx(np.log(new_v), self.mapping.v_log_edges)
                    # fill the transition matrix for this velocity series
                    fill_one_trajectory_sparse_with_freq_cython(lag, v_process_idx, new_f, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # fill the transition matrix for this angle series
                    theta_process_idx = self.mapping.find_1d_class_idx(new_theta, self.mapping.theta_edges)
                    fill_one_trajectory_sparse_with_freq_cython(lag, theta_process_idx, new_f, i_list_theta,
                                                                j_list_theta, ij_set_theta, val_list_theta)
        print 'done'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_v_class, n_v_class)), \
               csc_matrix((val_list_theta, (i_list_theta, j_list_theta)), shape = (n_theta_class, n_theta_class))

# TODO: add another class to derive the transition matrix for the case where the angle is a function of the
# TODO: previous angle and the distance from injection theta_n+1 = Z(theta_n, y_n)
class TransInfoIndependentVThetaY(object):
    """
    a class for extracting binned trasition information for 2d spatial cases with independent processes
    for the velocity p(v2|v1) and angle p(theta_2|theta_1, y_1) in time. The angle process depends on the distance
    from injection
    """
    def __init__(self, input_folder, n_total_realz, mapping, map_input, average_available=True,
                 time_step=None, raw_folder=None, train_time=None):
        """

        :param input_folder: the folder containing the particle tracking data
        :param n_binning_realz: number of realizations used for binning the data
        :param n_total_realz: total number of realizations
        :param n_absv_class: number of velocity classes
        :param n_theta_class: number of angle classes
        :param n_y_class: number of classes for the transverse distance from injection
        :param time_step: stencil time
        :param n_slow_class: number of slow classes (refining the velocity bins for very slow velocity values)
        :param max_allowed: max bins size allowed for log(v)
        """
        if not time_step:
            self.time_step = map_input.time_step
        self.raw_folder = input_folder
        if raw_folder:
            self.raw_folder = raw_folder
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        # self.make_velocity_bins = make_1d_abs_vel_bins
        self.mapping = mapping
        self.init_v_class_count, self.init_theta_class_count = self.get_init_class_count(map_input)
        self.average_available = average_available
        self.train_time = train_time

    def get_init_class_count(self, map_input):
        """
        :return:
         init_v_class_count: initial count of the velocity class. size (n_velocity_class,)
         init_v_theta_count: initial count of the angle class. size (n_theta_class,)
        """
        new_v, new_theta, new_f = remove_duplicate_xy(map_input.initial_v, map_input.initial_theta,
                                                      map_input.initial_f)
        init_v_idx = self.mapping.find_1d_class_idx(np.log(new_v), map_input.v_log_edges)
        # all the initial paths have zeros distance from injection
        init_thetaY_idx = self.mapping.class_index_2d_theta_y(new_theta, np.zeros(len(new_theta)))
        # initialize the count for each class
        init_v_class_count, init_thetaY_class_count = np.zeros(self.mapping.n_abs_v_classes), \
                                                      np.zeros(self.mapping.n_2d_theta_y_classes)
        for v_idx, theta_idx in zip(init_v_idx, init_thetaY_idx):
            init_v_class_count[v_idx] += 1
            init_thetaY_class_count[theta_idx] += 1
        return init_v_class_count, init_thetaY_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True, avg_available=None):
        if avg_available is None:
            avg_available = self.average_available
        if avg_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else:
            return self.get_trans_matrix_from_scratch(lag, print_every, verbose)

    def get_trans_matrix_from_average(self, lag, print_every=50, verbose=True):
        dt = self.time_step
        if self.train_time is not None:
            cut_off_train = int(self.train_time/dt)
        # get the size of the transition matrices
        n_v_class, n_theta_y_class = self.mapping.n_abs_v_classes, self.mapping.n_2d_theta_y_classes
        # initialize the sparse transition matrices
        print 'getting transition matrix from averaged data...'
        i_list_v, j_list_v, val_list_v = [[] for _ in range(3)]
        i_list_theta, j_list_theta, val_list_theta = [[] for _ in range(3)]
        ij_set_v, ij_set_theta = [set([]) for _ in range(2)]
        for j in range(self.n_total_realz):
            start_idx = 0
            # load the polar coordinates file
            data_path = os.path.join(self.input_folder, 'avg_polar_' + str(j) + '.npz')
            data = np.load(data_path)
            big_v, big_theta, big_f, ptr_list = data['V'], data['Theta'], data['F'], data['ptr']
            for i in ptr_list:
                new_v, new_theta, new_f = big_v[start_idx:i], big_theta[start_idx:i], big_f[start_idx:i]
                # adding in the repetitions because y is changing
                v_r = add_repetitions(new_v, new_f)
                theta_r = add_repetitions(new_theta, new_f)
                dy = np.multiply(v_r, np.sin(theta_r))*dt
                new_y = np.hstack((0.0, np.cumsum(dy)))[:-1]
                start_idx = i
                if len(new_v)>lag:
                    # simple process for v: v1, v2, v3, ...
                    v_process_idx = self.mapping.find_1d_class_idx(np.log(new_v), self.mapping.v_log_edges)
                    if self.train_time is not None:
                        # filter part of the trajectory
                        v_process_idx = v_process_idx[:cut_off_train]
                        new_f = new_f[:cut_off_train]
                    # fill the transition matrix for this velocity series
                    fill_one_trajectory_sparse_with_freq_cython(lag, v_process_idx, new_f, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # joint angle and y classes (theta1, y1), (theta2, y2), ...
                    # fill the transition matrix for this angle series
                    # joint_theta_process_idx = self.mapping.class_index_2d_theta_y(new_theta, new_y)
                    # fill_one_trajectory_sparse_with_freq_cython(lag, joint_theta_process_idx, new_f, i_list_theta,
                    #                                             j_list_theta, ij_set_theta, val_list_theta)
                    joint_theta_process_idx = self.mapping.class_index_2d_theta_y(theta_r, new_y)
                    fill_one_trajectory_sparse_cython(lag, joint_theta_process_idx, i_list_theta,
                                                      j_list_theta, ij_set_theta, val_list_theta)
        print 'done.'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_v_class, n_v_class)), \
               csc_matrix((val_list_theta, (i_list_theta, j_list_theta)), shape = (n_theta_y_class, n_theta_y_class))

    def get_trans_matrix_from_scratch(self, lag, print_every=50, verbose=True):
        """
        extract the transition matrix for velocity and angle for the given lag
        :param lag: the lag used to derive the transition matrix
        :return: v_trans_matrix, theta_trans_matrix for the given lag
        """
        # get the size of the transition matrices
        n_v_class, n_theta_y_class = self.mapping.n_abs_v_classes, self.mapping.n_2d_theta_y_classes
        # initialize the sparse transition matrices
        i_list_v, j_list_v, val_list_v = [[] for _ in range(3)]
        i_list_theta, j_list_theta, val_list_theta = [[] for _ in range(3)]
        ij_set_v, ij_set_theta = [set([]) for _ in range(2)]
        time_step = self.time_step
        input_folder = self.raw_folder
        print 'extracting trans matrix for the velocity and angle process...'
        for j in range(self.n_total_realz):
            if not j%5:
                print 'reading realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array) + 1e-15
            if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
                print 'some array was empty, skipping this file...'
                continue
            lastIdx = dataHolder.last_idx_array
            vxMatrix = np.divide(dx,dt)
            vyMatrix = np.divide(dy,dt)
            m = dx.shape[0]
            for i in range(m):
                x_start = dataHolder.x_array[i, 0]
                y_start = dataHolder.y_array[i, 0]
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start, time_step)
                dxTime, dyTime, freq = remove_duplicate_xy(dxTime, dyTime, freq)
                v_temp = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))/time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                # find y
                y_temp = np.hstack((0.0, np.cumsum(dyTime)))[:-1]
                if len(v_temp) > lag:
                    # simple process for v: v1, v2, v3, ...
                    v_process_idx = self.mapping.find_1d_class_idx(np.log(v_temp), self.mapping.v_log_edges)
                    # fill the transition matrix for this velocity series
                    fill_one_trajectory_sparse_with_freq_cython(lag, v_process_idx, freq, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # joint angle and y classes (theta1, y1), (theta2, y2), ...
                    # fill the transition matrix for this angle series
                    joint_theta_process_idx = self.mapping.class_index_2d_theta_y(theta_temp, y_temp)
                    fill_one_trajectory_sparse_with_freq_cython(lag, joint_theta_process_idx, freq, i_list_theta,
                                                                j_list_theta, ij_set_theta, val_list_theta)
        print 'done'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_v_class, n_v_class)), \
               csc_matrix((val_list_theta, (i_list_theta, j_list_theta)), shape = (n_theta_y_class, n_theta_y_class))