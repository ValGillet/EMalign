import numpy as np
import cv2

from cv2 import resize
from sofima import flow_field
from emprocess.utils.img_proc import downsample

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


def estimate_tilemap_overlap(tile_space,
                             tile_map,
                             preliminary_overlap=500,
                             scale=[0.3,0.5]):
    
    '''
    Estimate the overlap between tiles of a tile_map. 
    Using SIFT for estimate because more consistent results.
    '''
    
    offsets_x = []
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            left = tile_map[(x,y)] 
            right = tile_map[(x+1,y)] 

            try:
                offset, _ = estimate_transform_sift(left[:, -preliminary_overlap:], right[:, :preliminary_overlap], scale[0])
            except:
                offset, _ = estimate_transform_sift(left[:, -preliminary_overlap:], right[:, :preliminary_overlap], scale[1])
            offsets_x.append(offset[0])

    offsets_y = []
    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            bot = tile_map[(x,y)] 
            top = tile_map[(x,y+1)] 
            
            try:
                offset, _ = estimate_transform_sift(top[-preliminary_overlap:, :], bot[:preliminary_overlap, :], scale[0])
            except:
                offset, _ = estimate_transform_sift(top[-preliminary_overlap:, :], bot[:preliminary_overlap, :], scale[1])
            offsets_y.append(offset[1])

    overlap_x = preliminary_overlap - np.abs(offsets_x).max() if offsets_x else 0
    overlap_y = preliminary_overlap - np.abs(offsets_y).max() if offsets_y else 0
    
    return int(max(overlap_x, overlap_y))


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


def estimate_transform_sift(img1, img2, scale=1, return_homography=False):

    '''
    Estimate transformation (xy offset and rotation) from img2 to img1
    '''

    # Downsample images for faster computations
    ds_img1 = downsample(img1, scale)
    ds_img2 = downsample(img2, scale)

    # Find keypoints using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ds_img1,None)
    kp2, des2 = sift.detectAndCompute(ds_img2,None)

    # Match keypoints to each other
    # Brute force matchers is slower than flann, but it is exact
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Estimate affine transformation matrix
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Extract translation offsets
    xy_offset = M[:, 2]

    # Extract rotation angle in degrees
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    
    # Offset in pixels, rotation before transform
    if return_homography:
        return M, xy_offset/scale, theta
    else:
        return xy_offset/scale, theta