import logging
import numpy as np

from emprocess.utils.transform import rotate_image  

from .sift import estimate_transform_sift
from .pad import xy_offset_to_pad, pad_to_shape
from .utils import compute_laplacian_var_diff


def get_overlap(img1, 
                img2, 
                xy_offset, 
                rotation_angle):

    '''
    Extract overlapping parts of two images based on an offset and rotation from img2 to img1.
    '''

    # Estimate overlap
    # Masks and images are padded to same shape to facilitate comparison
    # I'm sure there is a smartest way but this works
    mask1 = np.ones_like(img1)
    mask1 = rotate_image(img1, -rotation_angle)
    img1 = rotate_image(img1, -rotation_angle)

    mask2 = np.ones_like(img2).astype(bool)
    img1=np.pad(img1, xy_offset_to_pad(-xy_offset))
    img2=np.pad(img2, xy_offset_to_pad(xy_offset))

    mask1=np.pad(mask1, xy_offset_to_pad(-xy_offset))
    mask2=np.pad(mask2, xy_offset_to_pad(xy_offset))

    max_shape = np.max([img1.shape, img2.shape], axis=0)

    img1 = pad_to_shape(img1, max_shape)
    img2 = pad_to_shape(img2, max_shape)
    mask1 = pad_to_shape(mask1, max_shape)
    mask2 = pad_to_shape(mask2, max_shape)
    
    mask = mask1.astype(bool) & mask2.astype(bool)

    if mask.any():
        y1,x1 = np.min(np.where(mask), axis=1) 
        y2,x2 = np.max(np.where(mask), axis=1) 

        return img1[y1:y2, x1:x2], img2[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    else:
        return None


def check_overlap(img1, 
                  img2, 
                  xy_offset, 
                  theta, 
                  threshold=0.5, 
                  scale=(0.3, 0.5), 
                  refine=True):

    '''
    Compute a metric describing how well images overlap, based on a given offset and rotation. 
    '''

    # Index of sharpness using Laplacian
    overlap = get_overlap(img1, img2, xy_offset, theta)

    if overlap is not None:
        overlap1, overlap2, mask = overlap

        lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)

        if refine and lap_variance_diff < threshold:
            logging.debug('Refining overlap estimation...')
            # Retry the overlap, it can often get better
            try:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[0])
            except:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[1])
            res = get_overlap(overlap1, overlap2, xy_offset, theta)
            
            if res is not None:
                overlap1, overlap2, mask = res
                lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)
            else:
                lap_variance_diff = 0
    else:
        # Images do not overlap (displacement is larger than image itself)
        lap_variance_diff = 0

    return lap_variance_diff
