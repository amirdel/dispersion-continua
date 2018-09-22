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
from py_dp.dispersion.independent_dispersion_models import DispModelIndependent
from py_dp.dispersion.trajectory_count_cython import get_cdf_cython

class DispModelIndependentVinsVThetaY(DispModelIndependent):
    """
    A class to model dispersion using two independent processes
    (v_avg, v_ins)
    (theta, Y)
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix_v, trans_matrix_theta, mapping, init_count_v,
                 init_count_theta, inj_location = "start", verbose = True):
        """

        :param n_particles: number of particles
        :param n_steps: number of steps
        :param dt: stencil time
        :param x_max: maximum length of the domain
        :param trans_matrix:
        :param mapping:
        :param init_class_count:
        :param inj_location:
        :param verbose:
        """
        super(DispModelIndependentVinsVThetaY,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                                        trans_matrix_theta, mapping, init_count_v, init_count_theta,
                                                        inj_location, verbose)
        # Since the classes are 2d, there are many more classes with zero initial count.
        # having a cdf for all the class counts will draw many classes that we did not have in the input data
        self.init_v_nz_idx = np.where(init_count_v>0)[0]
        self.init_theta_nz_idx = np.where(init_count_theta>0)[0]
        # get_cf_cython will change the input but we don't need the input anymore
        self.init_cdf_v_nz = get_cdf_cython(init_count_v[self.init_v_nz_idx])
        self.init_cdf_theta_nz = get_cdf_cython(init_count_theta[self.init_theta_nz_idx])
        self.closest_dict = self.closest_state_dictionary(trans_matrix_theta)


    def follow_all_particles_vector(self, verbose=True, print_every=50):
        """
        follow all the particles given the initial cdf's and the transition matrices
        :return:
        """
        dt = self.dt
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        v_mat = self.trans_matrix_v
        theta_mat = self.theta_mat_marginal
        # initialize indices for the velocity process from the initial cdf,
        # using only classes with non zero count
        v_2d_idx = np.array(np.searchsorted(self.init_cdf_v_nz, np.random.rand(n_particles)), dtype=np.int)
        v_2d_idx = self.init_v_nz_idx[v_2d_idx]
        v_idx, _ = self.mapping.class_index_1d_v_v_inst_from_2d(v_2d_idx)
        # initialize indices for the joint angle-Y process from the initial cdf
        # using only classes with non zero count
        theta_y_idx = np.array(np.searchsorted(self.init_cdf_theta_nz, np.random.rand(n_particles)), dtype=np.int)
        theta_y_idx = self.init_theta_nz_idx[theta_y_idx]
        theta_idx, _ = self.mapping.class_index_1d_theta_y_from_2d(theta_y_idx)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # The way y is calculated, there are y's where (theta,Y) was not observed in training data
        closest_dict = self.closest_dict
        trans_mat_coo = theta_mat.tocoo()
        observed_classes = trans_mat_coo.col
        #loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            if verbose and not i%print_every:
                print 'step number: ',i
            # draw values for the average velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            # save the new x, y, t
            x_array[:, i+1] = np.multiply(v_array, np.cos(theta_array))*dt + x_array[:, i]
            y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt + y_array[:, i]
            t_array[:, i+1] = dt + t_array[:, i]
            # choose the next 2d velocity class
            next_v_2d_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_2d_idx)
            if -12 in set(next_v_2d_idx):
                print 'number of weird v 2d class: ', len(np.where(next_v_2d_idx==-12)[0])
            # update v_idx
            v_idx, _ = self.mapping.class_index_1d_v_v_inst_from_2d(next_v_2d_idx)
            # update the velocity state
            v_2d_idx = np.copy(next_v_2d_idx)
            # update the joint angle-y class given the new y value
            new_y = np.reshape(y_array[:,i+1], theta_array.shape)
            theta_y_idx = self.mapping.class_index_2d_theta_y(theta_array, new_y)
            theta_y_idx = self.closest_observed(theta_y_idx, observed_classes, closest_dict)
            # choose the next joint angle-y class
            next_theta_y_idx = self.choose_next_class_vector(theta_mat.indptr, theta_mat.indices, theta_mat.data,
                                                             theta_y_idx)
            if -12 in set(next_theta_y_idx):
                print 'number of weird theta 2d class: ', len(np.where(next_theta_y_idx==-12)[0])
            # update theta_idx
            theta_idx, _ = self.mapping.class_index_1d_theta_y_from_2d(next_theta_y_idx)
            # remove the paths that were discontinued

            keep_mask[next_v_2d_idx==-12] = False
            keep_mask[next_theta_y_idx == -12] = False

            v_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
            print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]
