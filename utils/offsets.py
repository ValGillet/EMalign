import numpy as np
import cv2

from cv2 import resize
from sofima import flow_field
from emprocess.utils.img_proc import downsample

from emalign.utils.arrays import *


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


def estimate_transform_sift(img1, img2, scale=1, refine_estimate=True):

    '''
    Estimate transformation (xy offset and rotation) from img2 to img1
    Returns offset in pixels, rotation before transform, whether estimate is robust based on number and proportion of matches used
    '''

    # knnMatch will return an error if there are too many keypoints so we limit their number
    # see
    max_features=250000

    # Downsample images for faster computations
    ds_img1 = downsample(img1, scale)
    ds_img2 = downsample(img2, scale)

    # Find keypoints using SIFT
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(ds_img1,None)
    kp2, des2 = sift.detectAndCompute(ds_img2,None)

    # Match keypoints to each other
    # Brute force matchers is slower than flann, but it is exact
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Estimate affine transformation matrix
    try:
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    except cv2.error as e:
        if 'count >= 0' in e.err:
            M = None
        else:
            raise e
    except Exception as e:
        raise e
               
    if M is not None:
        # Extract translation offsets, divide by scale so it matches initial image resolution
        xy_offset = M[:, 2] / scale

        # Extract rotation angle in degrees
        theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

        robust_estimate = (len(good_matches)>10) and (inliers.sum() / len(good_matches) > 0.5)
    else:
        xy_offset = None
        theta = None
        robust_estimate = False

    if refine_estimate and not robust_estimate and scale<0.9:
        return estimate_transform_sift(img1, img2, scale=scale+0.1, refine_estimate=False)
    else:
        return xy_offset, theta, robust_estimate


def estimate_rough_z_offset(img1, img2, scale=0.1, range_limit=10, filter_size=5, patch_factor=1): 
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
    mask_1 = pad_to_shape(mask_1, target_shape, pad_value=1)
    mask_2 = pad_to_shape(mask_2, target_shape, pad_value=1)

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    xo, yo, _, pr = mfc.flow_field(
        img1_ds, img2_ds, 
        pre_mask=mask_1, post_mask=mask_2, 
        patch_size=tuple((np.array(img1_ds.shape)/patch_factor).astype(int).tolist()), 
        step=tuple((np.array(img1_ds.shape)/patch_factor).astype(int).tolist())
    ).squeeze()

    return np.array([yo, xo])/scale, pr