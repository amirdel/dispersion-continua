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
from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit
from py_dp.dispersion.trajectory_count_cython import get_cdf_cython
import bisect as bs


def moments_given_pdf(x, pdf_x):
    # make sure sum(pdf_x)*dx = 1
    dx = x[1] - x[0]
    pdf_copy = pdf_x[:]
    pdf_copy /= (sum(pdf_copy)*dx)
    mean = np.sum(x*pdf_copy)*dx
    var = np.sum(dx*(x**2)*pdf_copy) - mean**2
    skew = np.sum(dx*pdf_copy*(x-mean)**3)/var**1.5
    kurt = np.sum(dx*pdf_copy*(x-mean)**4)/var**2
    return mean, var, skew, kurt


def trans_prob_given_y(marginal, y_class, mapping):
    n_theta_class = mapping.n_theta_classes
    theta_trans_mat = np.zeros((n_theta_class, n_theta_class))
    all_theta_class = np.array(range(mapping.n_theta_classes))
    # the columns corresponding to that y
    cols = all_theta_class*mapping.n_y_classes + y_class
    data = marginal.data
    indices = marginal.indices
    indptr = marginal.indptr
    for idx, col in enumerate(cols):
        start, end = indptr[col], indptr[col+1]
        rows = indices[start:end]
        vals = data[start:end]
        theta_trans_mat[rows, all_theta_class[idx]] = vals
    return theta_trans_mat


def trans_prob_given_theta(marginal, theta_class, mapping):
    n_y_class = mapping.n_y_classes
    n_theta_class = mapping.n_theta_classes
    # our put has n_theta_class rows and n_y_class columns
    theta_trans_mat = np.zeros((n_theta_class, n_y_class))  # the output
    all_y_class = np.array(range(mapping.n_y_classes))
    # the columns corresponding to that theta
    cols = theta_class*mapping.n_y_classes + all_y_class
    data = marginal.data
    indices = marginal.indices
    indptr = marginal.indptr
    for idx, col in enumerate(cols):
        start, end = indptr[col], indptr[col+1]
        rows = indices[start:end]
        vals = data[start:end]
        theta_trans_mat[rows, all_y_class[idx]] = vals
    return theta_trans_mat


class MomentsModel:
    def __init__(self, marginal, mapping, mid_theta, mid_y, query_theta,
                 query_y, theta_range, y_range):
        self.marginal = marginal
        self.mapping = mapping
        self.mid_theta = mid_theta
        self.mid_y = mid_y
        self.query_theta = query_theta  # theta columns for which we derive trans mat
        self.theta_range = theta_range
        self.query_y = query_y
        self.y_range = y_range
        self.mean_slopes, self.mean_intercepts, self.var_dict = self.mean_var_model()
        self.skew_slopes, self.skew_intercepts = self.skew_model()
        self.pre_process()

    def pre_process(self):
        # make sure there is symmetry in variance
        # make sure there is symmetry in mean
        # assuming odd number of bins with zero in the center!
        sorted_keys = sorted(self.mean_slopes.keys())
        l, r = 0, len(sorted_keys) - 1
        mid_idx = r/2
        self.mean_intercepts[sorted_keys[mid_idx]] = 0.0
        while l < r:
            key_l, key_r = sorted_keys[l], sorted_keys[r]
            lv, rv = self.mean_slopes[key_l], self.mean_slopes[key_r]
            mean_val = 0.5 * (lv + rv)
            self.mean_slopes[key_l] = mean_val
            self.mean_slopes[key_r] = mean_val
            lv, rv = self.mean_intercepts[key_l], self.mean_intercepts[key_r]
            mean_val = 0.5*(abs(lv) + abs(rv))
            self.mean_intercepts[key_l] = -mean_val
            self.mean_intercepts[key_r] = mean_val
            lv, rv = self.var_dict[key_l], self.var_dict[key_r]
            mean_val = 0.5 * (lv + rv)
            self.var_dict[key_l] = mean_val
            self.var_dict[key_r] = mean_val
            l += 1
            r -= 1
            # use average skew slope for all models

    def mean_var_model(self):
        '''
        a model for the mean of theta_1 | theta_0, y_0
        mean model is a(theta)*y + b(theta). For theta in mid_theta[query_theta]
        we will save a, b in mean_slopes[theta], mean_intercepts[theta]
        '''
        y_array = self.mid_y[self.y_range]
        query_theta = self.query_theta
        mean_slope_dict, mean_intercept_dict, var_dict = {}, {}, {}
        for theta_class in query_theta:
            theta = self.mid_theta[theta_class]
            theta_trans_mat = trans_prob_given_theta(self.marginal, theta_class, self.mapping)
            moments_mat = np.zeros((theta_trans_mat.shape[1], 4))
            for j in self.y_range:
                prob = theta_trans_mat[:, j]
                moments_mat[j, :] = moments_given_pdf(self.mid_theta, prob)
            xp, yp = y_array, moments_mat[self.y_range, 0]
            nan_filter = ~np.isnan(yp)
            xp, yp = xp[nan_filter], yp[nan_filter]
            mean_slope_dict[theta], mean_intercept_dict[theta] = sci.polyfit(xp, yp, 1)
            var_vector = moments_mat[self.y_range, 1]
            var_vector = var_vector[~np.isnan(var_vector)]
            var_dict[theta] = np.mean(var_vector)
        return mean_slope_dict, mean_intercept_dict, var_dict

    def skew_model(self):
        '''
        a model for skewness of theta_1 | theta_0, y_0
        skew model is a(y)*theta_0 + b(y)
        '''
        theta_array = self.mid_theta[self.theta_range]
        query_y = self.query_y
        skew_slope_dict, skew_intercept_dict = {}, {}
        x_agg, y_agg = [], []
        for y_class in query_y:
            y = self.mid_y[y_class]
            theta_trans_mat = trans_prob_given_y(self.marginal, y_class, self.mapping)
            moments_mat = np.zeros((theta_trans_mat.shape[1], 4))
            for j in self.theta_range:
                prob = theta_trans_mat[:, j]
                moments_mat[j, :] = moments_given_pdf(self.mid_theta, prob)
            xp, yp = theta_array, moments_mat[self.theta_range, 2]
            nan_filter = ~np.isnan(yp)
            xp, yp = xp[nan_filter], yp[nan_filter]
            x_agg.extend(xp)
            y_agg.extend(yp)
            skew_slope_dict[y], skew_intercept_dict[y] = sci.polyfit(xp, yp, 1)
        line = lambda x,a: a*x
        popt, pcov = curve_fit(line, x_agg, y_agg)
        self.avg_skew_slope = popt[0]
        return skew_slope_dict, skew_intercept_dict


class DispModelComplex(DispModelIndependent):
    """
    A class to model dispersion using two independent processes
    (v_avg, v_ins)
    (theta, Y)
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix_v, trans_matrix_theta, mapping,
                 init_count_v, init_count_theta, var_coeff=1.0, inj_location = "start", verbose = True,
                 theta_diffusion_std=None):
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
        super(DispModelComplex,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                              trans_matrix_theta, mapping, init_count_v, init_count_theta,
                                              inj_location, verbose)
        self.theta_mat_marginal = self.theta_marginal_matrix()
        self.theta_mat_marginal.indptr = np.array(self.theta_mat_marginal.indptr, dtype=np.int)
        self.theta_mat_marginal.indices = np.array(self.theta_mat_marginal.indices, dtype=np.int)
        self.theta_mat_marginal.data = np.array(self.theta_mat_marginal.data, dtype=np.float)
        self.var_coeff = var_coeff
        self.theta_diffusion_std = theta_diffusion_std

    @staticmethod
    def interpolate(data_dict, query_array):
        sorted_keys = sorted(data_dict.keys())
        sorted_vals = [data_dict[key] for key in sorted_keys]
        return np.interp(query_array, sorted_keys, sorted_vals)

    def theta_marginal_matrix(self):
        """
        Get P(theta2 | theta1, y1) from P(theta2, y2 | theta1, y1)
        :return:
            The marginal theta transition matrix
        """
        mapping = self.mapping
        theta_mat_coo = self.trans_matrix_theta.tocoo()
        row, col, val = theta_mat_coo.row, theta_mat_coo.col, theta_mat_coo.data
        row_theta, row_y = mapping.class_index_1d_theta_y_from_2d(row)
        theta_conditional_mat = csc_matrix((val, (row_theta, col)),
                                           shape=(mapping.n_theta_classes, mapping.n_2d_theta_y_classes))
        return theta_conditional_mat

    def predict_theta(self, theta0, y0, moment_model):
        """
        Given theta0, y0 predict the next angle
        :param theta0:
        :param y0:
        :return:
        """
        dist_name = 'pearson3'
        dist = getattr(sci.stats, dist_name)
        mm = moment_model
        # mean parameters
        mean_slope_array = self.interpolate(mm.mean_slopes, theta0)
        mean_intercept_array = self.interpolate(mm.mean_intercepts, theta0)
        mean_array = mean_slope_array * y0 + mean_intercept_array
        # variance parameters
        var_array = self.interpolate(mm.var_dict, theta0)
        var_array = np.sqrt(var_array)
        # skew parameters
        # skew_slope_array = self.interpolate(mm.skew_slopes, y0)
        # skew_intercept_array = self.interpolate(mm.skew_intercepts, y0)
        # skew_array = skew_slope_array * theta0 + skew_intercept_array
        skew_array = mm.avg_skew_slope*theta0
        # draw theta
        next_theta = np.zeros(len(theta0))
        for i in range(len(theta0)):
            rv = dist(skew_array[i], mean_array[i], self.var_coeff * var_array[i] ** 0.5)
            next_theta[i] = rv.rvs()
        return next_theta

    def follow_all_particles_vector(self, moment_model, verbose=True, print_every=1):
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
        v_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
        # initialize indices for the joint angle-Y process from the initial cdf
        theta_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)),
                               dtype=np.int)
        theta_array = self.draw_from_class_theta(theta_idx)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose and not i % print_every:
                print 'step number: ', i
            # draw values for the velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            y_velocity = np.multiply(v_array, np.sin(theta_array)) * dt + y_array[:, i]
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i + 1] = np.multiply(v_array, np.cos(theta_array)) * dt + x_array[:, i]
            y_array[:, i + 1] = y_velocity
            t_array[:, i + 1] = dt + t_array[:, i]
            # choose the next velocity class
            next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
            if -12 in set(next_v_idx):
                print 'number of weird v class: ', len(np.where(next_v_idx == -12)[0])
            # update the joint angle-y class given the new y value
            new_y = np.reshape(y_array[:, i + 1], theta_array.shape)
            # choose the next angle class based on p(theta1 | theta0, y0)
            next_theta_array = self.predict_theta(theta_array, new_y, moment_model)
            theta_array = next_theta_array[:]
            if self.theta_diffusion_std is not None:
                theta_array += np.random.normal(0, self.theta_diffusion_std, len(theta_array))
            # remove the paths that were discontinued
            keep_mask[next_v_idx == -12] = False
            # update the idx arrays
            v_idx = next_v_idx
            v_idx[~keep_mask] = 0
        print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]