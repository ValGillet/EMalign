import numpy as np

from scipy import ndimage


def compute_mask(data, filter_size, range_limit):
    mask = (ndimage.maximum_filter(data, filter_size) 
            - ndimage.minimum_filter(data, filter_size)
            ) < range_limit
    return mask


def pad_to_shape(array, target_shape):
    assert array.ndim == len(target_shape)

    # Cannot pad with negative values
    end_pad = np.max([[0,0], target_shape-np.array(array.shape)], axis=0)
    pad = np.stack([(0,0), end_pad]).T
    return np.pad(array, pad)