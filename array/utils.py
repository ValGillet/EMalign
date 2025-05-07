import cv2
import numpy as np


# PAD
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


# ASSESS QUALITY
def _compute_laplacian_var(arr, mask=None):
    '''
    Compute laplacian variance. Provides an indication of image sharpness but is sensitive to contrast.
    '''
    if mask is not None:
        l = cv2.Laplacian(arr, cv2.CV_64F)[mask]
    else:
        l = cv2.Laplacian(arr, cv2.CV_64F)
    return np.var(l)


def _compute_sobel_mean(arr, mask=None):
    '''
    Apply Sobel operator. Provides an indication of image sharpness
    '''
    if mask is not None:
        sobel_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)[mask]
        sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)[mask]
    else:
        sobel_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(sobel)


def _compute_grad_mag(arr, mask=None):
    '''
    Compute gradient magnitude. Provides an indication of pixel variation, which can be used to measure sharpness/quality. 
    '''
    gy, gx = np.gradient(arr)

    if mask is not None:
        gnorm = np.sqrt(gx**2 + gy**2)[mask]
    else:
        gnorm = np.sqrt(gx**2 + gy**2)
    return np.average(gnorm)


def compute_laplacian_var_diff(overlap1, 
                               overlap2, 
                               mask=None):

    '''
    Compute a metric ([0,1]) describing how well two arrays overlap, based on laplacian filter.
    If score is 1, overlapping regions have the same edge content and therefore overlap well.
    '''
    
    lap_var1 = _compute_laplacian_var(overlap1, mask)
    lap_var2 = _compute_laplacian_var(overlap2, mask)

    # Calculate an index of difference in edge content (variance of laplacian)
    # Between 0 and 1, low means exact same content, 1 means different
    return 1 - abs(lap_var1 - lap_var2) / max(lap_var1, lap_var2)