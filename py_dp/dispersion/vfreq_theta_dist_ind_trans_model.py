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
import os
from scipy.sparse import csc_matrix
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_cython
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy
from py_dp.dispersion.average_trajectories import add_repetitions
from py_dp.dispersion.transition_matrix_fcns import normalize_columns


class TransInfoIndependentVFreqThetaDist(object):
    """
    a class for extracting binned trasition information for 2d spatial cases with independent processes
    for the velocity p(v2, f2|v1, f1) and angle p(theta|y) in time. The angle process depends on the distance
    from injection
    """
    def __init__(self, input_folder, n_total_realz, mapping, map_input, average_available=True,
                 time_step=None, raw_folder=None):
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
        self.init_vf_class_count, self.init_theta_class_count = self.get_init_class_count(map_input)
        self.average_available = average_available

    def get_init_class_count(self, map_input):
        """
        :return:
         init_v_class_count: initial count of the velocity class. size (n_velocity_class,)
         init_v_theta_count: initial count of the angle class. size (n_theta_class,)
        """
        # new_v, new_theta, new_f = remove_duplicate_xy(map_input.initial_v, map_input.initial_theta,
        #                                               map_input.initial_f)
        new_v, new_theta, new_f = map_input.initial_v, map_input.initial_theta, map_input.initial_f
        init_vf_idx = self.mapping.find_2d_class_idx_v_freq(new_v, new_f)
        # all the initial paths have zeros distance from injection
        init_theta_idx = self.mapping.find_1d_class_idx(new_theta, map_input.theta_edges)
        # initialize the count for each class
        init_vf_class_count, init_theta_class_count = np.zeros(self.mapping.n_vf_classes), \
                                                      np.zeros(self.mapping.n_theta_classes)
        for v_idx, theta_idx in zip(init_vf_idx, init_theta_idx):
            init_vf_class_count[v_idx] += 1
            init_theta_class_count[theta_idx] += 1
        return init_vf_class_count, init_theta_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True, avg_available=None):
        if avg_available is None:
            avg_available = self.average_available
        if avg_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else:
            raise('Average data does not exist!')

    def get_trans_matrix_from_average(self, lag, print_every=50, verbose=True):
        dt = self.time_step
        mapping = self.mapping
        # get the size of the transition matrices
        n_v_class, n_theta_class, n_y_class = mapping.n_abs_v_classes, mapping.n_theta_classes, \
                                                mapping.n_y_classes
        n_vf_class = mapping.n_vf_classes
        # initialize the sparse transition matrices
        print 'getting transition matrix from averaged data...'
        i_list_v, j_list_v, val_list_v = [[] for _ in range(3)]
        ij_set_v = set([])
        # initialize the theta matrix, each column is the distribution of theta at a given y
        theta_mat = np.zeros((n_theta_class, n_y_class))
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
                    vf_process_idx = self.mapping.find_2d_class_idx_v_freq(new_v, new_f)
                    # fill the transition matrix for this velocity-freq series
                    fill_one_trajectory_sparse_cython(lag, vf_process_idx, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # fill the angle matrix
                    y_class = mapping.find_1d_class_idx(new_y, mapping.y_edges)
                    theta_class = mapping.find_1d_class_idx(theta_r, mapping.theta_edges)
                    for yi, thetai in zip(y_class, theta_class):
                        theta_mat[thetai, yi] += 1
        theta_mat = normalize_columns(theta_mat)
        print 'done.'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_vf_class, n_vf_class)), theta_mat
