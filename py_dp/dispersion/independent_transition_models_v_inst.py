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
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_with_freq_cython
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xyv


class TransInfoIndependentVinstVThetaY(object):
    """
    Two processes:
    (v_avg, v_inst), (theta, y)
    """
    def __init__(self, input_folder, n_total_realz, mapping, map_input, average_available=True,
                 time_step=None):
        if not time_step:
            self.time_step = map_input.time_step
        self.raw_folder = input_folder
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        # self.make_velocity_bins = make_1d_abs_vel_bins
        self.mapping = mapping
        self.init_v_v_init_class_count, self.init_theta_y_class_count = self.get_init_class_count(map_input)
        self.average_available = average_available

    def get_init_class_count(self, map_input):
        """
        :return:
         init_v_v_init_class_count: initial count of the velocity class. size (n_velocity_class,)
         init_v_theta_count: initial count of the angle class. size (n_theta_class,)
        """
        new_v, new_theta, new_v_inst, new_f = remove_duplicate_xyv(map_input.initial_v, map_input.initial_theta,
                                                                   map_input.initial_v_inst, map_input.initial_f)
        init_v_idx = self.mapping.class_index_2d_v_v_inst(new_v, new_v_inst)
        # all the initial paths have zeros distance from injection
        init_thetaY_idx = self.mapping.class_index_2d_theta_y(new_theta, np.zeros(len(new_theta)))
        # initialize the count for each class
        init_v_v_init_class_count, init_thetaY_class_count = np.zeros(self.mapping.n_2d_v_v_inst_classes), \
                                                             np.zeros(self.mapping.n_2d_theta_y_classes)
        for v_idx, theta_idx in zip(init_v_idx, init_thetaY_idx):
            init_v_v_init_class_count[v_idx] += 1
            init_thetaY_class_count[theta_idx] += 1
        return init_v_v_init_class_count, init_thetaY_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True, avg_available=None):
        if avg_available is None:
            avg_available = self.average_available
        if avg_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else: raise('not implemented!')

    def get_trans_matrix_from_average(self, lag, print_every=50, verbose=True):
        dt = self.time_step
        # get the size of the transition matrices
        n_v_class, n_theta_y_class = self.mapping.n_2d_v_v_inst_classes, self.mapping.n_2d_theta_y_classes
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
            big_v, big_v_inst, big_theta, big_f, ptr_list = data['V'], data['V_inst'], \
                                                            data['Theta'], data['F'], data['ptr']
            for i in ptr_list:
                new_v, new_v_inst, new_theta, new_f = big_v[start_idx:i], big_v_inst[start_idx:i], \
                                                      big_theta[start_idx:i], big_f[start_idx:i]
                dy = np.multiply(new_v, np.sin(new_theta))*dt
                new_y = np.hstack((0.0, np.cumsum(dy)))[:-1]
                start_idx = i
                if len(new_v)>lag:
                    # simple process for v: v1, v2, v3, ...
                    v_process_idx = self.mapping.class_index_2d_v_v_inst(new_v, new_v_inst)
                    # fill the transition matrix for this velocity series
                    fill_one_trajectory_sparse_with_freq_cython(lag, v_process_idx, new_f, i_list_v, j_list_v,
                                                                ij_set_v, val_list_v)
                    # joint angle and y classes (theta1, y1), (theta2, y2), ...
                    # fill the transition matrix for this angle series
                    joint_theta_process_idx = self.mapping.class_index_2d_theta_y(new_theta, new_y)
                    fill_one_trajectory_sparse_with_freq_cython(lag, joint_theta_process_idx, new_f, i_list_theta,
                                                                j_list_theta, ij_set_theta, val_list_theta)
        print 'done.'
        return csc_matrix((val_list_v, (i_list_v, j_list_v)), shape = (n_v_class, n_v_class)), \
               csc_matrix((val_list_theta, (i_list_theta, j_list_theta)),
                          shape = (n_theta_y_class, n_theta_y_class))