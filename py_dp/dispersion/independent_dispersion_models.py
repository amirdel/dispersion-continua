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
from py_dp.dispersion.binning import get_cdf
from py_dp.dispersion.trajectory_count_cython import choose_next_class_vector_cython
from py_dp.dispersion.dispersion_models import dispersionModelGeneral
from scipy.sparse import csc_matrix
from py_dp.dispersion.transition_matrix_fcns import normalize_columns

class DispModelIndependent(dispersionModelGeneral):
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
        super(DispModelIndependent,self).__init__(n_particles, n_steps, inj_location, verbose)
        self.y_array = np.zeros(self.x_array.shape)
        self.trans_matrix_v = trans_matrix_v
        self.trans_matrix_theta = trans_matrix_theta
        self.fix_matrix_types()
        self.mapping = mapping
        self.dt = dt
        self.x_max = x_max
        # cdf for initial classes for whatever state definition being used
        self.init_class_cdf_v = get_cdf(init_count_v)
        self.init_class_cdf_theta = get_cdf(init_count_theta)
        self.blocked_particles = []
        self._switched_classes = {}
        print 'Normalizing transition matrices...'
        self.normalize_trans_mat_colums(self.trans_matrix_theta)
        self.normalize_trans_mat_colums(self.trans_matrix_v)


    def normalize_trans_mat_colums(self, trans_mat):
        for i in range(trans_mat.shape[1]):
            colsum = np.sum(trans_mat[:, i])
            if colsum > 0:
                trans_mat[:, i] /= colsum

    def fix_matrix_types(self):
        """
        change the type of the transition matrices to make it compatible with the cython code
        :return: No return values
        """
        self.trans_matrix_v.eliminate_zeros()
        self.trans_matrix_v.indptr = np.array(self.trans_matrix_v.indptr, dtype=np.int)
        self.trans_matrix_v.indices = np.array(self.trans_matrix_v.indices, dtype=np.int)
        self.trans_matrix_v.data = np.array(self.trans_matrix_v.data, dtype=np.float)
        if hasattr(self.trans_matrix_theta, 'indptr'):
            self.trans_matrix_theta.eliminate_zeros()
            self.trans_matrix_theta.indptr = np.array(self.trans_matrix_theta.indptr, dtype=np.int)
            self.trans_matrix_theta.indices = np.array(self.trans_matrix_theta.indices, dtype=np.int)
            self.trans_matrix_theta.data = np.array(self.trans_matrix_theta.data, dtype=np.float)

    def draw_from_class_velocity(self, idx, v_log_edges):
        """
        draw a vector of velocity values given a vector of velocity class indices
        :param idx: a vector of states
        :return: a vector of velocity values
        """
        x = np.random.rand(len(idx))
        log_v = np.multiply(v_log_edges[idx],x) + np.multiply(v_log_edges[idx + 1] ,1 - x)
        return np.exp(log_v)

    def draw_from_class_theta(self, idx):
        """
        draw a vector of angle values given a vector of angle states
        :param idx: a vector of angle states
        :return: a vector of angle values
        """
        x = np.random.rand(len(idx))
        theta_edges = self.mapping.theta_edges
        theta = np.multiply(theta_edges[idx], x) + np.multiply(theta_edges[idx + 1], 1 - x)
        return theta

    def draw_from_class_y(self, idx):
        x = np.random.rand(len(idx))
        y_edges = self.mapping.y_edges
        y = np.multiply(y_edges[idx], x) + np.multiply(y_edges[idx + 1], 1 - x)
        return y

    def choose_next_class_vector(self, indptr, indices, data, current_class):
        """
        function to choose the next class based on the current state and the sparse transition matrix
        :param indptr: pointer array for the beginning of each row
        :param indices: indices array
        :param data: data array
        :param current_class: current state
        :return: next state for this chain
        """
        return choose_next_class_vector_cython(indptr, indices, data, current_class)

    def closest_state_dictionary(self, transmat):
        """
        Create a map for every class to the closest class observed in the training data
        :param transmat: csc transition matrix
        :return: closest_dict: map between every state and the closest observed state
        """
        mapping = self.mapping
        n_class =transmat.shape[0]
        possible_2d_idx = np.array(range(n_class))
        theta_centers = mapping.theta_edges[:-1] + 0.5 * np.diff(mapping.theta_edges)
        y_centers = mapping.y_edges[:-1] + 0.5 * np.diff(mapping.y_edges)
        query_theta_idx, query_y_idx = mapping.class_index_1d_theta_y_from_2d(possible_2d_idx)
        query_theta = theta_centers[query_theta_idx]
        query_y = y_centers[query_y_idx]
        theta_mat_coo = transmat.tocoo()
        available_classes = np.unique(theta_mat_coo.col)
        target_theta_idx, target_y_idx = mapping.class_index_1d_theta_y_from_2d(available_classes)
        target_theta = theta_centers[target_theta_idx]
        target_y = y_centers[target_y_idx]
        X = np.vstack((query_y, query_theta)).T
        X_train = np.vstack((target_y, target_theta)).T
        # distance of every query class to every seen class, shape: (n_total, n_observed)
        dists = np.sqrt(-2 * np.dot(X, X_train.T) + \
                        np.sum(np.square(X)[:, np.newaxis, :], axis=2) + \
                        np.sum(np.square(X_train), axis=1))
        # index of closest member of seen classes to the query class
        idx = np.argmin(dists, axis=1)
        closest_dict = dict(zip(possible_2d_idx, available_classes[idx]))
        return closest_dict

    def find_closest_observed(self, query_idx_array, observed_idx_array, closest_dict):
        output_idx_array = np.copy(query_idx_array)
        not_seen_mask = np.in1d(query_idx_array, list(observed_idx_array), invert=True)
        closest_idx = np.array([closest_dict[i] for i in query_idx_array])
        output_idx_array[not_seen_mask] = closest_idx[not_seen_mask]
        # see which classes we switched to most
        for i in closest_idx[not_seen_mask]:
            self._switched_classes[i] = self._switched_classes.get(i, 0) + 1
        return output_idx_array


class DispModelIndependentSimple(DispModelIndependent):
    """
    A class to model dispersion using two independent processes for velocity and angle
    p(v_n+1|v_n) and p(theta_n+1|theta_n)
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
        super(DispModelIndependentSimple,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                                        trans_matrix_theta, mapping, init_count_v, init_count_theta,
                                                        inj_location, verbose)


    def follow_all_particles_vector(self, verbose=True):
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
        theta_mat = self.trans_matrix_theta
        # initialize indices for the velocity process from the initial cdf
        v_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
        # initialize indices for the angle process from the initial cdf
        theta_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)), dtype=np.int)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        #loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose:
                print 'step number: ',i
            # draw values for the velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i+1] = np.multiply(v_array, np.cos(theta_array))*dt
            y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt
            t_array[:, i+1] = dt
            # choose the next velocity and angle class
            next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
            next_theta_idx = self.choose_next_class_vector(theta_mat.indptr, theta_mat.indices, theta_mat.data, theta_idx)
            # remove the paths that were discontinued
            keep_mask[next_v_idx==-12] = False
            keep_mask[next_theta_idx == -12] = False
            # update the idx arrays
            v_idx, theta_idx = next_v_idx, next_theta_idx
            v_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
        x_array = x_array[keep_mask, :]
        self.x_array = np.cumsum(x_array, axis=1)
        y_array = y_array[keep_mask, :]
        self.y_array = np.cumsum(y_array, axis=1)
        t_array = t_array[keep_mask, :]
        self.time_array = np.cumsum(t_array, axis=1)


class DispModelIndependentVThetaY(DispModelIndependent):
    """
    A class to model dispersion using two independent processes for velocity and angle
    p(v_n+1|v_n) and p(theta_n+1|theta_n)
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix_v, trans_matrix_theta, mapping, init_count_v,
                 init_count_theta, inj_location = "start", verbose = True, k=5, distance='euclidean',
                 get_knn_marginal=False):
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
        :param k: k-nearest neighbors parameter
        """
        super(DispModelIndependentVThetaY,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix_v,
                                                        trans_matrix_theta, mapping, init_count_v, init_count_theta,
                                                        inj_location, verbose)
        self.closest_dict = self.closest_state_dictionary(trans_matrix_theta)
        self.theta_mat_marginal = self.theta_marginal_matrix()
        # fix types
        self.theta_mat_marginal.indptr = np.array(self.theta_mat_marginal.indptr, dtype=np.int)
        self.theta_mat_marginal.indices = np.array(self.theta_mat_marginal.indices, dtype=np.int)
        self.theta_mat_marginal.data = np.array(self.theta_mat_marginal.data, dtype=np.float)
        if get_knn_marginal:
            self.marginal_knn = self.generate_knn_marginal(k, distance)
            self.fix_type_single_matrix(self.marginal_knn)

    def fix_type_single_matrix(self, matrix):
        matrix.indptr = np.array(matrix.indptr, dtype=np.int)
        matrix.indices = np.array(matrix.indices, dtype=np.int)
        matrix.data = np.array(matrix.data, dtype=np.float)

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

    def generate_knn_marginal(self, k, distance):
        print '****Calculating knn smoothed marginal****'
        mapping = self.mapping
        marginal = self.theta_mat_marginal
        n_y_classes = mapping.n_y_classes
        theta_centers = mapping.theta_edges[:-1] + 0.5 * np.diff(mapping.theta_edges)
        y_centers = mapping.y_edges[:-1] + 0.5 * np.diff(mapping.y_edges)
        # find all the classes visited in extracting the transition matrix
        # these are the target y and theta
        theta_mat_coo = self.trans_matrix_theta.tocoo()
        # target classes are the ones with non-empty cdf
        target_classes = np.unique(theta_mat_coo.col)
        target_theta_idx, target_y_idx = mapping.class_index_1d_theta_y_from_2d(target_classes)
        target_theta = theta_centers[target_theta_idx]
        target_y = y_centers[target_y_idx]
        # all_y_classes = np.arange(n_y_classes)
        # interesting_theta_classes: classes where theta is observed
        # interesting_theta_classes = []
        # for tidx in np.unique(target_theta_idx):
        #     interesting_theta_classes.extend(tidx * np.ones(n_y_classes) * n_y_classes + all_y_classes)
        # # print 'len interesting: ', len(interesting_theta_classes)
        # # from these classes find the ones where (theta,y) is not observed
        # n_data = np.diff(marginal.indptr)
        # all_2d_classes = np.arange(mapping.n_2d_theta_y_classes)
        # observed_pairs = all_2d_classes[n_data > 0]
        # query_classes = np.setdiff1d(interesting_theta_classes, observed_pairs)
        alternative_query = np.setdiff1d(np.arange(mapping.n_2d_theta_y_classes), target_classes)
        # TODO:
        query_classes = alternative_query
        print 'len query classes for knn: ', len(query_classes)
        tqidx, tyidx = mapping.class_index_1d_theta_y_from_2d(query_classes)
        tqidx = np.array(tqidx, dtype=np.int)
        tyidx = np.array(tyidx, dtype=np.int)
        query_theta = theta_centers[tqidx]
        query_y = y_centers[tyidx]
        # define a distance function
        # option1: choose a theta as close as possible


        ## option2: Eclidean distance
        if distance == 'euclidean':
            X = np.vstack((query_y, query_theta)).T
            X_train = np.vstack((target_y, target_theta)).T
            dist = np.sqrt(-2 * np.dot(X, X_train.T) + np.sum(np.square(X)[:, np.newaxis, :], axis=2) +
                           np.sum(np.square(X_train), axis=1))
        else:
            t_coeff = 2.0e3
            y_coeff = 1.0
            dist = y_coeff * (query_y[np.newaxis].T - target_y) ** 2 + \
                   t_coeff * (query_theta[np.newaxis].T - target_theta) ** 2
        knn_class = np.argsort(dist, axis=1)[:, :k]
        knn_marginal = marginal.copy()
        # build the marginal knn columns by averaging the neighbors
        for q_idx, q in enumerate(query_classes):
            ngh_idx_in_target = knn_class[q_idx, :]
            knn_marginal[:, q] = np.sum(knn_marginal[:, target_classes[ngh_idx_in_target]], axis=1) / k
        return knn_marginal

    def follow_all_particles_vector_knn(self, verbose=True, print_every=1):
        """
        follow all the particles given the initial cdf's and the transition matrices
        :return:
            saves the plume evolution in x_array, y_array, time_array attributes
        """
        print '****Using vector knn for particle tracking****'
        dt = self.dt
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        v_mat = self.trans_matrix_v
        theta_mat_marginal = self.marginal_knn
        # initialize indices for the velocity process from the initial cdf
        v_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
        # initialize indices for the joint angle-Y process from the initial cdf
        theta_y_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)),
                               dtype=np.int)
        theta_idx, y_idx = self.mapping.class_index_1d_theta_y_from_2d(theta_y_idx)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # The way y is calculated, there are y's where (theta,Y) was not observed in training data
        # theta_mat = self.trans_matrix_theta
        # trans_mat_coo = theta_mat.tocoo()
        # observed_classes = trans_mat_coo.col
        # loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose and not i % print_every:
                print 'step number: ', i
            # draw values for the velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            # theta_rnd = 0.1*np.random.normal(size=len(theta_array))
            # theta_array += theta_rnd
            y_markov = self.draw_from_class_y(y_idx)
            y_velocity = np.multiply(v_array, np.sin(theta_array)) * dt + y_array[:, i]
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i + 1] = np.multiply(v_array, np.cos(theta_array)) * dt + x_array[:, i]
            # y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt + y_array[:, i]
            # y_array[:, i + 1] = 0.5*(y_markov + y_velocity)
            y_array[:, i + 1] = y_velocity
            t_array[:, i + 1] = dt + t_array[:, i]
            # choose the next velocity class
            next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
            if -12 in set(next_v_idx):
                print 'number of weird v class: ', len(np.where(next_v_idx == -12)[0])
            # update the joint angle-y class given the new y value
            input_y = np.reshape(y_array[:, i], theta_array.shape)
            # for the new y find the closest theta s.t. the class (theta, y) has been visited once
            theta_y_idx = self.mapping.class_index_2d_theta_y(theta_array, input_y)
            # theta_y_idx = self.closest_observed(theta_y_idx, observed_classes, closest_dict)
            # choose the next theta class given the current thetaY
            # TODO
            next_theta_idx = self.choose_next_class_vector(theta_mat_marginal.indptr, theta_mat_marginal.indices,
                                                           theta_mat_marginal.data, theta_y_idx)
            if -12 in set(next_theta_idx):
                print 'number of weird theta 2d class: ', len(np.where(next_theta_idx == -12)[0])
            # remove the paths that were discontinued
            keep_mask[next_v_idx == -12] = False
            keep_mask[next_theta_idx == -12] = False
            # update the idx arrays
            v_idx, theta_idx = next_v_idx, next_theta_idx
            v_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
        print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]

    def follow_all_particles_vector_marginal(self, verbose=True, print_every=5):
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
        theta_mat = self.trans_matrix_theta
        theta_mat_marginal = self.theta_mat_marginal
        # initialize indices for the velocity process from the initial cdf
        v_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
        # initialize indices for the joint angle-Y process from the initial cdf
        theta_y_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)),
                               dtype=np.int)
        theta_idx, y_idx = self.mapping.class_index_1d_theta_y_from_2d(theta_y_idx)
        keep_mask = np.ones(n_particles, dtype=bool)
        v_log_edges = self.mapping.v_log_edges
        # The way y is calculated, there are y's where (theta,Y) was not observed in training data
        closest_dict = self.closest_dict
        trans_mat_coo = theta_mat.tocoo()
        observed_classes = np.unique(trans_mat_coo.col)
        # loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            # if not i%50 and verbose:
            if verbose and not i % print_every:
                print 'step number: ', i
            # draw values for the velocity and angle processes given their class indices
            v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
            theta_array = self.draw_from_class_theta(theta_idx)
            # theta_rnd = 0.1*np.random.normal(size=len(theta_array))
            # theta_array += theta_rnd
            # y_markov = self.draw_from_class_y(y_idx)
            y_velocity = np.multiply(v_array, np.sin(theta_array)) * dt + y_array[:, i]
            # save the new dx, dy, dt to be integrated outside the loop
            x_array[:, i + 1] = np.multiply(v_array, np.cos(theta_array)) * dt + x_array[:, i]
            # y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt + y_array[:, i]
            # y_array[:, i + 1] = 0.5*(y_markov + y_velocity)
            y_array[:, i + 1] = y_velocity
            t_array[:, i + 1] = dt + t_array[:, i]
            # choose the next velocity class
            next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
            if -12 in set(next_v_idx):
                print 'number of weird v class: ', len(np.where(next_v_idx == -12)[0])
            # update the joint angle-y class given the new y value
            input_y = np.reshape(y_array[:, i], theta_array.shape)
            # for the new y find the closest theta s.t. the class (theta, y) has been visited once
            theta_y_idx = self.mapping.class_index_2d_theta_y(theta_array, input_y)
            theta_y_idx = self.find_closest_observed(theta_y_idx, observed_classes, closest_dict)
            # choose the next theta class given the current thetaY
            # TODO
            next_theta_idx = self.choose_next_class_vector(theta_mat_marginal.indptr, theta_mat_marginal.indices,
                                                           theta_mat_marginal.data, theta_y_idx)
            if -12 in set(next_theta_idx):
                print 'number of weird theta 2d class: ', len(np.where(next_theta_idx == -12)[0])
            # remove the paths that were discontinued
            keep_mask[next_v_idx == -12] = False
            keep_mask[next_theta_idx == -12] = False
            # update the idx arrays
            v_idx, theta_idx = next_v_idx, next_theta_idx
            v_idx[~keep_mask] = 0
            theta_idx[~keep_mask] = 0
        print '***number of discarded particles:', len(keep_mask[~keep_mask])
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]

    # def follow_all_particles_vector(self, verbose=True, print_every=50):
    #     """
    #     follow all the particles given the initial cdf's and the transition matrices
    #     :return:
    #         saves the plume evolution in x_array, y_array, time_array attributes
    #     """
    #     dt = self.dt
    #     x_array = self.x_array
    #     y_array = self.y_array
    #     t_array = self.time_array
    #     n_particles = self.n_particles
    #     v_mat = self.trans_matrix_v
    #     theta_mat = self.trans_matrix_theta
    #     # initialize indices for the velocity process from the initial cdf
    #     v_idx = np.array(np.searchsorted(self.init_class_cdf_v, np.random.rand(n_particles)), dtype=np.int)
    #     # initialize indices for the joint angle-Y process from the initial cdf
    #     theta_y_idx = np.array(np.searchsorted(self.init_class_cdf_theta, np.random.rand(n_particles)), dtype=np.int)
    #     theta_idx, y_idx = self.mapping.class_index_1d_theta_y_from_2d(theta_y_idx)
    #     keep_mask = np.ones(n_particles, dtype=bool)
    #     v_log_edges = self.mapping.v_log_edges
    #     # The way y is calculated, there are y's where (theta,Y) was not observed in training data
    #     # TODO: use k-nearest neighbors here!
    #     closest_dict = self.closest_dict
    #     trans_mat_coo = theta_mat.tocoo()
    #     observed_classes = trans_mat_coo.col
    #     # loop over all steps and save dx, dy, dt
    #     for i in range(self.n_steps):
    #         # if not i%50 and verbose:
    #         if verbose and not i%print_every:
    #             print 'step number: ',i
    #         # draw values for the velocity and angle processes given their class indices
    #         v_array = self.draw_from_class_velocity(v_idx, v_log_edges)
    #         theta_array = self.draw_from_class_theta(theta_idx)
    #         y_markov = self.draw_from_class_y(y_idx)
    #         y_velocity = np.multiply(v_array, np.sin(theta_array))*dt + y_array[:, i]
    #         # save the new dx, dy, dt to be integrated outside the loop
    #         x_array[:, i+1] = np.multiply(v_array, np.cos(theta_array))*dt + x_array[:, i]
    #         # y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt + y_array[:, i]
    #         # y_array[:, i + 1] = 0.5*(y_markov + y_velocity)
    #         y_array[:, i + 1] = y_velocity
    #         t_array[:, i+1] = dt + t_array[:, i]
    #         # choose the next velocity class
    #         next_v_idx = self.choose_next_class_vector(v_mat.indptr, v_mat.indices, v_mat.data, v_idx)
    #         if -12 in set(next_v_idx):
    #             print 'number of weird v class: ', len(np.where(next_v_idx==-12)[0])
    #         # update the joint angle-y class given the new y value
    #         new_y = np.reshape(y_array[:,i+1], theta_array.shape)
    #         # for the new y find the closest theta s.t. the class (theta, y) has been visited once
    #         theta_y_idx = self.mapping.class_index_2d_theta_y(theta_array, new_y)
    #         theta_y_idx = self.closest_observed(theta_y_idx, observed_classes, closest_dict)
    #         # choose the next joint angle-y class
    #         next_theta_y_idx = self.choose_next_class_vector(theta_mat.indptr, theta_mat.indices, theta_mat.data, theta_y_idx)
    #         if -12 in set(next_theta_y_idx):
    #             print 'number of weird theta 2d class: ', len(np.where(next_theta_y_idx==-12)[0])
    #         next_theta_idx, y_idx = self.mapping.class_index_1d_theta_y_from_2d(next_theta_y_idx)
    #         # remove the paths that were discontinued
    #         keep_mask[next_v_idx==-12] = False
    #         keep_mask[next_theta_y_idx == -12] = False
    #         # update the idx arrays
    #         v_idx, theta_idx = next_v_idx, next_theta_idx
    #         v_idx[~keep_mask] = 0
    #         theta_idx[~keep_mask] = 0
    #     print '***number of discarded particles:', len(keep_mask[~keep_mask])
    #     x_array = x_array[keep_mask, :]
    #     y_array = y_array[keep_mask, :]
    #     t_array = t_array[keep_mask, :]