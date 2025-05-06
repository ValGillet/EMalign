import cv2
import logging
import numpy as np


def xy_offset_to_pad(offset):
    pad = np.zeros([2,2], dtype=int)
    x,y = [int(i) for i in offset]
    
    if y > 0:
        pad[0][1] = y
    else:
        pad[0][0] = abs(y)
    
    if x > 0:
        pad[1][1] = x
    else:
        pad[1][0] = abs(x)

    return pad


def pad_to_shape(array, 
                 target_shape, 
                 direction=[1,1], 
                 pad_value=0):
    '''
    Pad an array to match a shape. If the target shape is smaller than the array's shape in a dimension, no padding is added to that dimension.
    '''
    assert array.ndim == len(target_shape)

    pad_size = target_shape-np.array(array.shape)
    pad_size[pad_size<0] = 0

    pad = np.zeros([2,2]).astype(int)
    for i, d in enumerate(direction):
        d = max(0, d)
        pad[i][d] = pad_size[i]
    return np.pad(array, pad, constant_values=pad_value)


def homogenize_arrays_shape(arrs):
    max_shape = np.max([a.shape for a in arrs],axis=0)
    return [pad_to_shape(a, max_shape) for a in arrs]


def compute_laplacian_var_diff(overlap_1, 
                               overlap_2, 
                               mask):

    '''
    Compute a metric ([0,1]) describing how well two arrays overlap, based on laplacian filter.
    If score is 1, overlapping regions have the same edge content and therefore overlap well.
    '''
    
    mask = np.ones_like(overlap_1).astype(bool) if mask is None else mask

    laplacian1 = cv2.Laplacian(overlap_1, cv2.CV_64F)[mask]
    laplacian2 = cv2.Laplacian(overlap_2, cv2.CV_64F)[mask]

    lap_var1 = np.var(laplacian1)
    lap_var2 = np.var(laplacian2)

    # Calculate an index of difference in edge content (variance of laplacian)
    # Between 0 and 1, low means exact same content, 1 means different
    return 1 - abs(lap_var1 - lap_var2) / max(lap_var1, lap_var2)