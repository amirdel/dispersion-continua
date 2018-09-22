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

from copy import copy
import numpy as np
import bisect as bs
from py_dp.dispersion.binning import fix_bisect_left_indices

# TODO: move to a better place
def make_auxillary_arrays(n_base_class_idx, base_class_idx_sample, f_sample):
    """
    This function makes auxilary variables used for mapping for classes where the second dimension
    is the number of repetitions for the first dimension (base_class_idx, f)
    :param n_base_class_idx: number of classes for the base class (e.g. you can have 100 velocity classes)
    :param base_class_idx_sample: observed sample for base class
    :param f_sample: observed frequency for the sample
    :return:
    sub_classes_nrepeat: array of arrays, each small array is the observed f values for one class of the base variable,
                         for each primary class at list one subclass with repetition of one will be considered.
    cumsum_n_subclasses: This is the cumulative sum of the number of subclasses.
    """
    base_class_idx_sample = np.array(base_class_idx_sample, dtype=np.int)
    print 'making auxillary arrays for mapping...'
    sub_classes_nrepeat = []
    n_subclass = []
    place_holder = np.array([1], dtype=np.int)
    # loop over v-theta classes
    for i in range(n_base_class_idx):
        possible_f_vals = np.unique(f_sample[base_class_idx_sample == i])
        if not len(possible_f_vals):
            possible_f_vals = copy(place_holder)
        sub_classes_nrepeat.append(sorted(possible_f_vals))
        n_subclass.append(len(possible_f_vals))
    modified_n_sub_class = np.array(n_subclass)
    cumsum_n_subclass = np.hstack((0, np.cumsum(modified_n_sub_class)))
    print 'done'
    return sub_classes_nrepeat, cumsum_n_subclass

def find_extended_class_number(primary_index_array, freq_array, sub_classes_nrepeat, cumsum_n_subclass):
    """
    convert (primary_idx, f) to a single extended index
    :param primary_index_array:
    :param freq_array:
    :param sub_classes_nrepeat:
    :param cumsum_n_subclass:
    :return:
    """
    assert(len(primary_index_array) == len(freq_array))
    sub_classes_nrepeat = sub_classes_nrepeat
    cumsum_n_subclass = cumsum_n_subclass
    extended_index_array = np.zeros(len(primary_index_array), dtype=np.int)
    for i in range(len(primary_index_array)):
        class_2d = primary_index_array[i]
        freq = freq_array[i]
        sub_class_array = sub_classes_nrepeat[class_2d]
        ind_repeat = np.array(bs.bisect_left(sub_class_array, freq))
        # check out of bounds
        fix_bisect_left_indices(ind_repeat, sub_class_array)
        extended_index_array[i] = cumsum_n_subclass[class_2d] + ind_repeat
    return extended_index_array

def find_primary_index(extended_index, n_extended_classes, n_primary_classes,
                       sub_classes_nrepeat, cumsum_n_subclass):
    """
    convert extended index to (primary_index, f)
    :param self:
    :param extended_index:
    :param sub_classes_nrepeat:
    :param cumsum_n_subclass:
    :return:
    """
    if np.any(extended_index >= n_extended_classes):
        raise ('index of extended class can not be larger than number of classes - ')
    extended_index = np.array(extended_index, dtype=np.int)
    last_idx_array = cumsum_n_subclass[1:] - 1
    primary_index = np.searchsorted(last_idx_array, extended_index)
    # check out of bounds
    primary_index -= (primary_index == n_primary_classes)
    # freq_idx is the idx of freq in the freq_array containing all possible frequencies
    # for a given extended idx
    freq_idx = extended_index - cumsum_n_subclass[primary_index]
    freq = np.zeros(len(extended_index), dtype=np.int)
    for i, primary_idx in enumerate(primary_index):
        freq_array = sub_classes_nrepeat[primary_idx]
        freq[i] = freq_array[freq_idx[i]]
    return primary_index, freq