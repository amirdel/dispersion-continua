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
from scipy.optimize import leastsq
from py_dp.dispersion.trajectory_count_cython import get_cdf_cython
import bisect as bs

def moments_given_theta_mat_all(theta_mat, mid_theta):
    dtheta = mid_theta[1] - mid_theta[0]
    theta_mat_copy = np.copy(theta_mat)
    # make sure the transition matrix is normalized
    colsum = np.sum(theta_mat_copy, axis=0)
    colsum[colsum == 0] = 1.0
    theta_mat_copy /= colsum
    theta_mat_copy /= dtheta
    mean_array = []
    var_array = []
    skew_array = []
    kurt_array = []
    not_none_idx = []
    sm = 1e-14
    for col in range(theta_mat_copy.shape[1]):
        prob = theta_mat_copy[:, col]
        mean, var, skew, kurt = [None for _ in range(4)]
        # if non-zero entry
        if np.any(prob):
            mean = np.sum(mid_theta*prob)*dtheta
            var = np.sum(dtheta*(mid_theta**2)*prob) - mean**2
            skew = np.sum(dtheta*prob*(mid_theta-mean)**3)/(var+sm)**1.5
            kurt = np.sum(dtheta*prob*(mid_theta-mean)**4)/(var+sm)**2
            not_none_idx.append(col)
        var_array.append(var)
        mean_array.append(mean)
        skew_array.append(skew)
        kurt_array.append(kurt)
    kurt_array = np.array(kurt_array)
    kurt_array[not_none_idx] -= 3
    return np.array(mean_array), np.array(var_array), np.array(skew_array), np.array(kurt_array)


class DispModelIndependentVThetaDist(DispModelIndependent):
    """
    A class to model dispersion using two independent processes
    (v_avg, v_ins)
    (theta, Y)
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix_v, trans_matrix_theta, mapping, init_count_v,
                 init_count_theta, inj_location = "start", verbose = True, theta_var_coeff = 1.0, k=5,
                 dist_cutoff = 20, use_kmeans=False, model_type='pearson3', replace_center=False,
                 theta_base_coeff = 1.0, theta_diffusion=None, lsq=True, lsq_frac=0.25):
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
        super(DispModelIndependentVThetaDist,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                                        trans_matrix_theta, mapping, init_count_v, init_count_theta,
                                                        inj_location, verbose)
        self.not_seen_y = np.where(np.sum(self.trans_matrix_theta, 0) == 0.0)[0]
        self.closest_y_list = self.closest_seen_y()
        # need a solution for holes in the data
        if use_kmeans:
            print 'using kmeans smoothing'
            self.theta_mat_kmean_smoothed = self.theta_mat_kmeans(k, dist_cutoff)
            theta_mat = self.theta_mat_kmean_smoothed
        if lsq:
            print 'using modeled theta matrix fitted with least squares: ' + model_type
            self.lsq_frac = lsq_frac
            self.theta_mat_model = self.model_theta_mat_lsq(model_type)
            theta_mat = self.theta_mat_model
        else:
            print 'using modeled theta transition matrix: ' + model_type
            self.theta_mat_model = self.model_theta_mat(theta_var_coeff, model_type,
                                                        hybrid=replace_center, base_coeff=theta_base_coeff)
            theta_mat = self.theta_mat_model
        # make sure theta mat sums up to one
        colsum = np.sum(theta_mat, axis=0)
        colsum[colsum==0] = 1
        theta_mat /= colsum
        self.theta_mat_final = theta_mat
        self.theta_cdf = np.cumsum(theta_mat, axis=0)
        self.theta_diffusion_std = theta_diffusion

    def theta_mat_kmeans(self, k, cutoff):
        '''

        :param k: k-nearest neighbors to use for averaging
        :param cutoff: noisy class will be ones with less than 30 non-zero entries for the pdf
        :return:
            theta_mat_kmeans: trans_mat where the noisy columns have been replaced by the
                              the average of k nearest decent columns
        '''
        assert(isinstance(k, int))
        assert(k > 0)
        mapping = self.mapping
        trans_mat = self.trans_matrix_theta
        # make sure that transmat is normalized
        col_sum = np.sum(trans_mat, axis=0)
        col_sum[col_sum == 0] = 1.0
        trans_mat /= col_sum
        # theta_mat_kmeans = np.zeros((mapping.n_theta_classes, mapping.n_y_classes))
        theta_mat_kmeans = np.copy(trans_mat)
        y_centers = mapping.y_edges[:-1] + 0.5*np.diff(mapping.y_edges)
        # noisy_y_idx = np.where(np.abs(y_centers) > 0.8*y_centers[-1])[0]
        noisy_y_idx = np.where(np.sum(trans_mat > 0, axis=0) < cutoff)[0]
        decent_y_idx = np.setdiff1d(range(len(y_centers)), noisy_y_idx)
        # dist matrix: every row contains the distance of a noisy y to all not noisy y's
        dist = (y_centers[noisy_y_idx][np.newaxis].T - y_centers[decent_y_idx]) ** 2
        k_nearest = np.argsort(dist, axis=1)[:, :k]
        # replace the probability in the noisy columns by the average of k nearest goof columns
        theta_mat_kmeans[:, noisy_y_idx] = np.sum(trans_mat.T[decent_y_idx[k_nearest]], axis=1).T/float(k)
        return theta_mat_kmeans

    def model_theta_mat(self, theta_var_coeff, model_type='pearson3', hybrid=False, base_coeff = 1.0):
        '''
        model P(theta|y) given the moments from empirical distribution.
        Here least squares is not used for finding the optimal parameters! Part of the development of the idea...
        :param theta_var_coeff:
        :param model_type:
        :param hybrid:
        :param base_coeff:
        :return:
        '''
        theta_mat = self.trans_matrix_theta
        mapping = self.mapping
        n_y_classes = mapping.n_y_classes
        mid_theta = np.diff(mapping.theta_edges) / 2 + mapping.theta_edges[:-1]
        mid_y = np.diff(mapping.y_edges) / 2 + mapping.y_edges[:-1]
        # ss, ll = int(0.4*n_y_classes), int(0.6*n_y_classes)
        ss, ll = int(0.25 * n_y_classes), int(0.75 * n_y_classes)
        col_array = range(ss, ll+1)
        # get the moments from the transition matrix
        mean_array, var_array, skew_array, kurt_array = moments_given_theta_mat_all(theta_mat, mid_theta)
        ys = mid_y[col_array]
        key, val_array = [], []
        order_array = [1, 0, 1, 0]
        for idx, array in enumerate([mean_array[col_array], var_array[col_array],
                                     skew_array[col_array], kurt_array[col_array]]):
            order = order_array[idx]
            key.append(idx)
            fit = sci.polyfit(ys, array, order)
            val_array.append(fit)
        reg_dict = dict(zip(key, val_array))
        # model the theta matrix
        mean_from_model = sci.polyval(reg_dict[0], mid_y)
        var_from_model = sci.polyval(reg_dict[1], mid_y)
        skew_from_model = sci.polyval(reg_dict[2], mid_y)
        theta_mat_model = np.zeros((mapping.n_theta_classes, mapping.n_y_classes))
        if model_type == 'truncnorm':
            print 'inside model: truncnorm'
            for i in range(mapping.n_y_classes):
                myclip_a, myclip_b = -np.pi, np.pi
                my_mean, my_std = mean_from_model[i], (theta_var_coeff*var_from_model[i]) ** 0.5
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                theta_mat_model[:, i] = truncnorm.pdf(mid_theta, a, b, my_mean, my_std)
                theta_mat_model[:, i] /= np.sum(theta_mat_model[:, i])
        elif model_type == 'pearson3':
            print 'inside model: pearson3'
            dist_name = 'pearson3'
            pearson = getattr(sci.stats, dist_name)
            dd_array = []
            for i in range(mapping.n_y_classes):
                if i in col_array:
                    final_coeff = base_coeff
                    dd_array.append(base_coeff)
                elif i < ss:
                    slope = (theta_var_coeff - base_coeff) / (mid_y[0] - mid_y[ss])
                    dd = slope * (mid_y[i] - mid_y[ss]) + base_coeff
                    dd_array.append(dd)
                    final_coeff = dd
                elif i > ll:
                    slope = (theta_var_coeff - base_coeff) / (mid_y[-1] - mid_y[ll])
                    dd = slope * (mid_y[i] - mid_y[ll]) + base_coeff
                    dd_array.append(dd)
                    final_coeff = dd
                vari = var_from_model[i] * final_coeff
                theta_mat_model[:, i] = pearson.pdf(mid_theta, skew_from_model[i],
                                                         mean_from_model[i], vari ** 0.5)
                theta_mat_model[:, i] /= np.sum(theta_mat_model[:, i])
        else:
            raise('model type not implemented!')
        if hybrid:
            theta_mat_model[:, col_array] = theta_mat[:, col_array]
        self.dd = dd_array
        return theta_mat_model

    def model_theta_mat_lsq(self, model_type='pearson3', verbose=False):
        '''
        Actual function being used for filling the holes in P(theta|y)
        :param model_type:
        :return:
        '''
        theta_mat = self.trans_matrix_theta
        mapping = self.mapping
        n_y_classes = mapping.n_y_classes
        mid_theta = np.diff(mapping.theta_edges) / 2 + mapping.theta_edges[:-1]
        mid_y = np.diff(mapping.y_edges) / 2 + mapping.y_edges[:-1]
        #####################################################################################
        dtheta = mid_theta[1] - mid_theta[0]
        theta_mat_copy = np.copy(theta_mat)
        # make sure the transition matrix is normalized
        colsum = np.sum(theta_mat_copy, axis=0)
        colsum[colsum == 0] = 1.0
        theta_mat_copy /= colsum
        theta_mat_copy /= dtheta
        theta_mat = theta_mat_copy
        #####################################################################################
        f1, f2 = self.lsq_frac, 1.0-self.lsq_frac
        ss, ll = int(f1 * n_y_classes), int(f2 * n_y_classes)
        col_array = range(ss, ll + 1)
        self.col_array = col_array
        # get the moments from the transition matrix
        mean_array, var_array, skew_array, kurt_array = moments_given_theta_mat_all(theta_mat, mid_theta)
        self.discrete_moments = (mean_array, var_array, skew_array)
        ###############################################################################
        mean_array2, var_array2, skew_array2, kurt_array2 = [np.zeros(len(mid_y)) for _ in range(4)]

        def err_fcn(p, x, y, dist_name):
            return y - fit_fcn(p, dist_name, x)

        def fit_fcn(p, dist_name, x):
            dist = getattr(sci.stats, dist_name)
            return dist.pdf(x, *p)
        if verbose: print 'using pearson-lsq'
        dist_name = 'pearson3'
        dist = getattr(sci.stats, dist_name)
        # loop over all middle range y's and find the lsq fit to the histogram
        for col in col_array:
            prob = theta_mat[:, col]
            x, y = mid_theta, prob
            # use the discrete moments as an initial guess
            init = [skew_array[col], mean_array[col], var_array[col]]
            out = leastsq(err_fcn, init, xtol=1e-10, args=(x, y, dist_name))
            param = out[0]
            # print 'lsq param: ', param
            lsq_moments = dist.stats(*param, moments='mvsk')
            mean_array2[col], var_array2[col], skew_array2[col], kurt_array2[col] = lsq_moments[0], lsq_moments[1], \
                                                                                    lsq_moments[2], lsq_moments[3]
        self.lsq_moments = (mean_array2, var_array2, skew_array2)

        #################################################################################
        ys = mid_y[col_array]
        key, val_array = [], []
        order_array = [1, 0, 1, 0]
        for idx, array in enumerate([mean_array2[col_array], var_array2[col_array],
                                     skew_array2[col_array], kurt_array2[col_array]]):
            order = order_array[idx]
            key.append(idx)
            fit = sci.polyfit(ys, array, order)
            val_array.append(fit)
        reg_dict = dict(zip(key, val_array))
        # model the theta matrix
        mean_from_model = sci.polyval(reg_dict[0], mid_y)
        var_from_model = sci.polyval(reg_dict[1], mid_y)
        skew_from_model = sci.polyval(reg_dict[2], mid_y)
        self.fitted_moments = (mean_from_model, var_from_model, skew_from_model)
        theta_mat_model = np.zeros((mapping.n_theta_classes, mapping.n_y_classes))
        self.dd = np.ones(len(mid_y))
        if model_type == 'truncnorm':
            if verbose: print 'inside model: truncnorm'
            for i in range(mapping.n_y_classes):
                myclip_a, myclip_b = -np.pi, np.pi
                my_mean, my_std = mean_from_model[i], (var_from_model[i]) ** 0.5
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                theta_mat_model[:, i] = truncnorm.pdf(mid_theta, a, b, my_mean, my_std)
                theta_mat_model[:, i] /= np.sum(theta_mat_model[:, i])
        elif model_type == 'pearson3':
            if verbose: print 'inside model: pearson3'
            dist_name = 'pearson3'
            pearson = getattr(sci.stats, dist_name)
            for i in range(mapping.n_y_classes):
                vari = var_from_model[i]
                theta_mat_model[:, i] = pearson.pdf(mid_theta, skew_from_model[i],
                                                    mean_from_model[i], vari ** 0.5)
                theta_mat_model[:, i] /= np.sum(theta_mat_model[:, i])
        return theta_mat_model

    def closest_seen_y(self):
        """

        :return:
            closest_idx_list: closest_idx_list[i] contains the closest y class that has observed data
        """
        y_centers = np.diff(self.mapping.y_edges)/2 + self.mapping.y_edges[:-1]
        prob_sum = np.sum(self.trans_matrix_theta, 0)
        not_seen_idx = np.where(prob_sum == 0.0)[0]
        seen_idx = np.setdiff1d(range(len(y_centers)), not_seen_idx)
        # for every idx not observed in the data, find the closest observed class
        dist = np.abs(y_centers[not_seen_idx][np.newaxis].T - y_centers[seen_idx])
        closest_idx_list = np.array(range(len(y_centers)))
        closest_idx_for_unseen = seen_idx[np.argmin(dist, axis=1)]
        closest_idx_list[not_seen_idx] = closest_idx_for_unseen
        return closest_idx_list

    def choose_next_theta_given_y(self, y_class_array):
        """
        Given an array for the y class of each particle, draw a random angle index from the theta
        distribution for that y class
        :param y_class_array:
        :return:
            theta_class_array
        """
        # if there are no examples seen for a given y go to the closest y
        not_seen_y = np.in1d(y_class_array, self.not_seen_y)
        y_class_array[not_seen_y] = self.closest_y_list[y_class_array[not_seen_y]]
        rand_array = np.random.random(size=len(y_class_array))
        theta_cdf = self.theta_cdf
        theta_idx_array = np.array([np.searchsorted(theta_cdf[:, y_class_array[i]], rand_array[i])
                for i in range(len(y_class_array))])
        return theta_idx_array

    def follow_all_particles_vector(self, verbose=True, print_every=50, warn=True):
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
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # loop over all steps and save dx, dy, dt
        p_change = 1.0
        print 'running bernoulli with change state prob = ',p_change
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose and not i % print_every:
                print 'step number: ', i
            # draw values for the velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            if self.theta_diffusion_std is not None:
                theta_array += np.random.normal(0, self.theta_diffusion_std, len(theta_array))
            y_velocity = np.multiply(v_array, np.sin(theta_array)) * dt + y_array[:, i]
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i + 1] = np.multiply(v_array, np.cos(theta_array)) * dt + x_array[:, i]
            y_array[:, i + 1] = y_velocity
            t_array[:, i + 1] = dt + t_array[:, i]
            # choose the next velocity class
            next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
            if warn and -12 in set(next_v_idx):
                print 'number of weird v class: ', len(np.where(next_v_idx == -12)[0])
            # update the joint angle-y class given the new y value
            new_y = np.reshape(y_array[:, i + 1], theta_array.shape)
            # choose the next angle class based on the current y locations
            y_class_array = self.mapping.find_1d_class_idx(new_y, self.mapping.y_edges)
            next_theta_idx = self.choose_next_theta_given_y(y_class_array)
            # toss a coin and decide whether to change the theta class
            coin_toss = np.random.binomial(1, p_change, n_particles)
            keep_unchanged_mask = coin_toss<1
            next_theta_idx[keep_unchanged_mask] = theta_idx[keep_unchanged_mask]
            if warn and -12 in set(next_theta_idx):
                print 'number of weird theta 2d class: ', len(np.where(next_theta_idx == -12)[0])
            # remove the paths that were discontinued
            keep_mask[next_v_idx == -12] = False
            keep_mask[next_theta_idx == -12] = False
            # update the idx arrays
            v_idx, theta_idx = next_v_idx, next_theta_idx
            v_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
        if warn: print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]