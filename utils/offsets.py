import numpy as np

from cv2 import resize
from sofima import stitch_rigid, flow_field

from .arrays import *

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


def estimate_offset_vert(top, bot, overlap):
    top = top[-overlap:, :]
    bot = bot[:overlap, :]
    
    # Compensate for difference in shape
    shape_diff = np.array(top.shape) - np.array(bot.shape)
    if np.any(shape_diff > 0):
        bot = np.pad(bot, [(shape_diff[0], 0), (shape_diff[1], 0)])
    elif np.any(shape_diff < 0):
        top = np.pad(top, [(abs(shape_diff[0]), 0), (abs(shape_diff[1]), 0)])
    
    xy_offset, _ = stitch_rigid._estimate_offset(top, bot, 0, filter_size=5)
    return xy_offset, top, bot


def estimate_offset_horiz(left, right, overlap):
    left = left[:, -overlap:]
    right = right[:, :overlap]
    
    # Compensate for difference in shape
    shape_diff = np.array(left.shape) - np.array(right.shape)
    if np.any(shape_diff > 0):
        right = np.pad(right, [(0, shape_diff[0]), (0, shape_diff[1])])
    elif np.any(shape_diff < 0):
        left = np.pad(left, [(0, abs(shape_diff[0])), (0, abs(shape_diff[1]))])
    
    xy_offset, _ = stitch_rigid._estimate_offset(left, right, 0, filter_size=5)
    return xy_offset, left, right


def estimate_rough_z_offset(img1, img2, scale=0.1, range_limit=1, filter_size=5): 
    '''
    Based on sofima.stitch_rigid._estimate_offset
    '''
    img1_ds = resize(img1, None, fx=scale, fy=scale)
    img2_ds = resize(img2, None, fx=scale, fy=scale) 

    mask_1 = compute_mask(img1_ds, filter_size, range_limit)
    mask_2 = compute_mask(img2_ds, filter_size, range_limit)

    # Pad to the same shape to avoid errors with flow computation
    # Pad after computing the masks to save time
    target_shape = np.max([img1_ds.shape, img2_ds.shape], axis=0)

    img1_ds = pad_to_shape(img1_ds, target_shape)
    img2_ds = pad_to_shape(img2_ds, target_shape)
    mask_1 = pad_to_shape(mask_1, target_shape)
    mask_2 = pad_to_shape(mask_2, target_shape)

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    xo, yo, _, _ = mfc.flow_field(
        img1_ds, img2_ds, pre_mask=mask_1, post_mask=mask_2, patch_size=img1_ds.shape, step=(1, 1)
    ).squeeze()

    return np.array([yo, xo])/scale