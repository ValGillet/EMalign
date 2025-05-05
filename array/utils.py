import cv2
import logging
import numpy as np

from .pad import pad_to_shape


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