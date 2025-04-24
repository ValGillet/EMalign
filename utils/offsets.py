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
    try:
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)    
    except:
        raise RuntimeError('Error while estimating homography. There may not be enough keypoints.')
    if M is None:
        raise ValueError('Error while estimating homography. Keypoints may not correspond between images')

    # Extract translation offsets
    xy_offset = M[:, 2]

    # Extract rotation angle in degrees
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    
    # Offset in pixels, rotation before transform
    if return_homography:
        return M, xy_offset/scale, theta
    else:
        return xy_offset/scale, theta


def estimate_tiles_overlap(img1, 
                           img2, 
                           prelim_overlap,
                           axis, 
                           scale,
                           score_threshold=0.8):
    from emalign.utils.tile_map_positions import check_overlap
    
    if axis == 0:
        crop_img1 = img1[-prelim_overlap:, :]
        crop_img2 = img2[:prelim_overlap, :]
    elif axis == 1:
        crop_img1 = img1[:, -prelim_overlap:]
        crop_img2 = img2[:, :prelim_overlap]

    # First, try increasing only compute scale
    try:
        offset, angle = estimate_transform_sift(crop_img1, crop_img2, scale[0])
        overlap_score = check_overlap(crop_img1, crop_img2, offset, angle, threshold=score_threshold, refine=True)            
    except:
        # There may not be enough keypoints to compute offset
        overlap_score = 0

    if overlap_score < score_threshold:
        # Decrease downsampling
        try:
            offset, angle = estimate_transform_sift(crop_img1, crop_img2, scale[1])
            overlap_score = check_overlap(crop_img1, crop_img2, offset, angle, threshold=score_threshold, refine=True)
        except:
            overlap_score = 0
    
    if overlap_score > score_threshold:
        overlap = prelim_overlap - np.abs(offset[::-1][axis])
        return offset, overlap, overlap_score
    
    # Second, try increasing overlap
    max_overlap = max(img1.shape[axis], img2.shape[axis])
    prelim_overlap = max_overlap // 2

    if axis == 0:
        crop_img1 = img1[-prelim_overlap:, :]
        crop_img2 = img2[:prelim_overlap, :]
    elif axis == 1:
        crop_img1 = img1[:, -prelim_overlap:]
        crop_img2 = img2[:, :prelim_overlap]

    try:
        offset, angle = estimate_transform_sift(crop_img1, crop_img2, scale[0])
        overlap_score = check_overlap(crop_img1, crop_img2, offset, angle, threshold=score_threshold, refine=True)            
    except:
        # There may not be enough keypoints to compute offset
        overlap_score = 0

    if overlap_score < score_threshold:
        # Decrease downsampling
        try:
            offset, angle = estimate_transform_sift(crop_img1, crop_img2, scale[1])
            overlap_score = check_overlap(crop_img1, crop_img2, offset, angle, threshold=score_threshold, refine=True)
        except:
            overlap_score = 0
            offset = 0

    overlap = prelim_overlap - np.abs(offset[::-1][axis]) if overlap_score > score_threshold else 0
    return offset, overlap, overlap_score


def estimate_tilemap_overlap(tile_space,
                             tile_map,
                             prelim_overlap=500,
                             scale=[0.5,1],
                             score_threshold=0.8):
    
    '''
    Estimate the overlap between tiles of a tile_map. 
    Using SIFT for estimate because more consistent results.
    '''
    
    overlaps_x = []
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            left = tile_map[(x,y)] 
            right = tile_map[(x+1,y)] 

            _, overlap, _ = estimate_tiles_overlap(left, 
                                                       right, 
                                                       prelim_overlap=prelim_overlap,
                                                       axis=1, 
                                                       scale=scale,
                                                       score_threshold=score_threshold)
            overlaps_x.append(overlap)

    overlaps_y = []
    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            bot = tile_map[(x,y)] 
            top = tile_map[(x,y+1)] 
            
            _, overlap, _ = estimate_tiles_overlap(bot, 
                                                   top, 
                                                   prelim_overlap=prelim_overlap,
                                                   axis=0, 
                                                   scale=scale,
                                                   score_threshold=score_threshold)
            overlaps_y.append(overlap)
    return int(np.max(overlaps_x + overlaps_y))


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