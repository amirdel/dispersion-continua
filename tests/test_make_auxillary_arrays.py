from unittest import TestCase
import numpy as np
from py_dp.dispersion.mapping_aux_functions import make_auxillary_arrays, \
    find_extended_class_number, find_primary_index

class Test_Extended_auxillary_arrays(TestCase):
    def test_make_auxillary_arrays(self):
        n_base_class_idx = 5
        base_class_sample = np.array([0, 0, 1, 2, 2, 3, 4])
        f_sample = np.array([3, 1, 1, 2, 1, 1, 2])
        sub_class_nrepeat, cumsum_n_subclass = make_auxillary_arrays(n_base_class_idx, base_class_sample,
                                                                     f_sample)
        expected_sub_class_n_repeat = [[1, 3], [1], [1, 2], [1], [2]]
        expected_cumsum_n_subclass = [0, 2, 3, 5, 6, 7]
        np.testing.assert_equal(sub_class_nrepeat, expected_sub_class_n_repeat)
        np.testing.assert_equal(cumsum_n_subclass, expected_cumsum_n_subclass)

    def test_find_extended_index(self):
        n_base_class_idx = 5
        base_sample = np.array([0, 0, 1, 2, 2, 3, 4])
        freq_sample = np.array([3, 1, 1, 2, 1, 1, 2])
        sub_class_nrepeat, cumsum_n_subclass = make_auxillary_arrays(n_base_class_idx, base_sample,
                                                                     freq_sample)
        extended_index = find_extended_class_number(base_sample, freq_sample, sub_class_nrepeat,
                                                    cumsum_n_subclass)
        expected_extended_idx = [1, 0, 2, 4, 3, 5, 6]
        np.testing.assert_equal(extended_index, expected_extended_idx)

    def test_find_primary_index(self):
        n_base_class_idx = 5
        base_sample = np.array([0, 0, 1, 2, 2, 3, 4])
        freq_sample = np.array([3, 1, 1, 2, 1, 1, 2])
        sub_class_nrepeat, cumsum_n_subclass = make_auxillary_arrays(n_base_class_idx, base_sample,
                                                                     freq_sample)
        extended_idx = [1, 0, 2, 4, 3, 5, 6]
        n_extended_class = cumsum_n_subclass[-1]
        primary_index, f = find_primary_index(extended_idx, n_extended_class, n_base_class_idx,
                                              sub_class_nrepeat, cumsum_n_subclass)
        np.testing.assert_equal(primary_index, base_sample)
        np.testing.assert_equal(f, freq_sample)