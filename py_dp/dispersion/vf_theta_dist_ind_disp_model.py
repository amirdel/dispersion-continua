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
import scipy as sci
from scipy.stats import truncnorm
from py_dp.dispersion.independent_dispersion_models_theta_dist import moments_given_theta_mat_all, \
    DispModelIndependentVThetaDist


class DispModelVF_ThetaDist(DispModelIndependentVThetaDist):
    """
    A class to model dispersion using two independent processes
    (v_avg, v_ins)
    (theta, Y)
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix_v, trans_matrix_theta, mapping, init_count_v,
                 init_count_theta, inj_location = "start", verbose = True, theta_var_coeff = 1.0, k=5,
                 dist_cutoff = 20, use_kmeans=False, model_type='pearson3', replace_center=False,
                 theta_base_coeff = 1.0, theta_diffusion=None):
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
        super(DispModelVF_ThetaDist, self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                                    trans_matrix_theta, mapping, init_count_v,
                                                    init_count_theta, inj_location, verbose, theta_var_coeff,
                                                    k, dist_cutoff, use_kmeans, model_type, replace_center,
                                                    theta_base_coeff, theta_diffusion)

    def follow_all_particles_vector(self, verbose=True, print_every=50):
        """
        follow all the particles given the initial cdf's and the transition matrices
        :return:
            saves the plume evolution in x_array, y_array, time_array attributes
        """
        dt = self.dt
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        v_mat = self.trans_matrix_v
        # initialize indices for the velocity process from the initial cdf
        vf_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
        # initialize indices for the joint angle-Y process from the initial cdf
        theta_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)),
                               dtype=np.int)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # loop over all steps and save dx, dy, dt
        # p_change = 1.0
        # if p_change < 1.0:
        #     print 'running bernoulli with change state prob = ',p_change
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose and not i % print_every:
                print 'step number: ', i
            # draw values for the velocity and angle processes given their class indices
            v_idx, f_array = self.mapping.get_v_f_given_2d(vf_idx)
            # print 'number of f>1: ', f_array[f_array>1]
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            # move the particle given new v, theta, f
            y_velocity = np.multiply(np.multiply(v_array, np.sin(theta_array)), dt*f_array) + y_array[:, i]
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i + 1] = np.multiply(np.multiply(v_array, np.cos(theta_array)), dt*f_array) + x_array[:, i]
            y_array[:, i + 1] = y_velocity
            t_array[:, i + 1] = dt*f_array + t_array[:, i]
            # choose the next velocity class
            next_vf_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, vf_idx)
            if -12 in set(next_vf_idx):
                print 'number of weird v class: ', len(np.where(next_vf_idx == -12)[0])
            # update the joint angle-y class given the new y value
            new_y = np.reshape(y_array[:, i + 1], theta_array.shape)
            # choose the next angle class based on the current y locations
            y_class_array = self.mapping.find_1d_class_idx(new_y, self.mapping.y_edges)
            next_theta_idx = self.choose_next_theta_given_y(y_class_array)

            # # TODO: toss a coin and decide whether to change the theta class
            # coin_toss = np.random.binomial(1, p_change, n_particles)
            # keep_unchanged_mask = coin_toss<1
            # next_theta_idx[keep_unchanged_mask] = theta_idx[keep_unchanged_mask]

            if -12 in set(next_theta_idx):
                print 'number of weird theta 2d class: ', len(np.where(next_theta_idx == -12)[0])
            # remove the paths that were discontinued
            keep_mask[next_vf_idx == -12] = False
            keep_mask[next_theta_idx == -12] = False
            # update the idx arrays
            vf_idx, theta_idx = next_vf_idx, next_theta_idx
            vf_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
        print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]