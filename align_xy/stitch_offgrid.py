'''
Functions for stitching images existing on the same XY plane.
Contrary to tilemaps, these functions are for images that are not on a predictable grid. 
'''

import numpy as np
import jax.numpy as jnp

from connectomics.common import bounding_box
from scipy.ndimage import binary_erosion
from sofima import flow_field, flow_utils, mesh
from sofima.warp import ndimage_warp

from emprocess.utils.mask import compute_greyscale_mask, mask_to_bbox
from emprocess.utils.transform import rotate_image

from ..array.sift import estimate_transform_sift
from ..array.utils import homogenize_arrays_shape, xy_offset_to_pad, _compute_laplacian_var
from ..array.overlap import get_overlap
from .utils import mask_to_mesh


def get_elastic_mesh(pre, 
                     post, 
                     pre_mask, 
                     post_mask, 
                     patch_size, 
                     stride,
                     k0=0.01,
                     k=0.1,
                     gamma=0.5):
    
    # Estimate flow
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flow = mfc.flow_field(pre, post, (patch_size,patch_size),
                            (stride,stride), batch_size=256,
                            pre_mask=~pre_mask, 
                            post_mask=~post_mask)

    # Clean flow
    pad = patch_size // 2 // stride
    flow = np.pad(flow, [[0, 0], [pad, pad - 1], [pad, pad - 1]], constant_values=np.nan)
    kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, "max_deviation": 5, "max_magnitude": 0}
    flow = flow_utils.clean_flow(flow[:, None, ...], **kwargs)

    mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=gamma, k0=k0, k=k, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=False)
    x = mask_to_mesh(post_mask, stride)
    x, _, _ = mesh.relax_mesh(jnp.array(x), -flow, mesh_config)
    return np.array(x)[:,0,...]


def render_fused_slice(pre, 
                       post,
                       pre_mask,
                       post_mask,
                       x,
                       stride,
                       bbox_post=[0,-1,0,-1],
                       work_size=512,
                       overlap=1,
                       parallelism=1,
                       output_shape=None,
                       post_on_top=False,
                       resize_canvas=True
                       ):
    
    y1,y2,x1,x2 = bbox_post
    post = post[y1:y2, x1:x2]
    post_mask = post_mask[y1:y2, x1:x2]

    if output_shape is None:
        output_shape = post.shape

    data_bbox = bounding_box.BoundingBox(start=(0, 0), 
                                        size=(output_shape[-1], output_shape[-2]))
    # Warp image
    warped_img = ndimage_warp(
        post, 
        x, 
        stride=(stride, stride), 
        work_size=(work_size, work_size), 
        overlap=(overlap,overlap),
        image_box=data_bbox,
        parallelism=parallelism
    )

    # Warp mask
    warped_mask = ndimage_warp(
        post_mask, 
        x, 
        stride=(stride, stride),
        work_size=(work_size, work_size),
        overlap=(overlap,overlap),
        image_box=data_bbox,
        parallelism=parallelism
    )

    # Stitch images together
    stitched = pre.copy()
    if post_on_top:
        m = warped_mask
    else:
        m = warped_mask & ~pre_mask[y1:y2,x1:x2]

    stitched[y1:y2,x1:x2][m] = warped_img[m]
    stitched_mask = pre_mask
    stitched_mask[y1:y2,x1:x2] |= warped_mask
    
    if resize_canvas:
        y1,y2,x1,x2 = mask_to_bbox(stitched_mask)
        stitched = stitched[y1:y2,x1:x2]
        stitched_mask = stitched_mask[y1:y2,x1:x2]
    return stitched, stitched_mask


def stitch_images(img1, 
                  img2,
                  mask1=None, 
                  mask2=None,
                  scale=0.1,
                  patch_size=160,
                  stride=40,
                  parallelism=1,
                  img_on_top='auto',
                  img_q_fun=_compute_laplacian_var,
                  resize_canvas=True,
                  **kwargs):
    
    '''
    Stitch two images on the same slice. Img1 is the reference, img2 is the moving image.

    img_q_fun: function taking image and mask as arguments, returns a value higher for higher quality/sharpness.
    e.g.: img_q_fun = lambda img, m: _compute_laplacian_var(img, m)*0.5 + _compute_sobel_mean(img, m) + _compute_grad_mag(img, m)*100
    '''

    # Compute mask if not provided
    mask1 = mask1.astype(bool) if mask1 is not None else compute_greyscale_mask(img1)
    mask2 = mask2.astype(bool) if mask2 is not None else compute_greyscale_mask(img2)

    # Estimate rigid offset
    offset, rotation, _ = estimate_transform_sift(img1, img2, scale)   
    
    # Refine estimate for a better final result
    o1, o2 = get_overlap(img1, img2, offset, rotation)   
    offset2, _, _ = estimate_transform_sift(o1, o2, scale)

    if img_on_top == 'auto':
        # Determine automatically what image to put on top
        m1, m2 = get_overlap(mask1, mask2, offset, rotation)
        l1 = img_q_fun(o1, m1)
        l2 = img_q_fun(o2, m2)
        if l1 > l2:
            post_on_top = False
        else:
            post_on_top = True
    elif img_on_top == '1':
        post_on_top = False
    elif img_on_top == '2':
        post_on_top = True

    offset += offset2

    # Rotate image
    img2 = rotate_image(img2, rotation)
    mask2 = rotate_image(mask2.astype(np.uint8), rotation).astype(bool)
  
    # Pad images and masks to apply the offset
    img1 = np.pad(img1, xy_offset_to_pad(-offset))
    img2 = np.pad(img2, xy_offset_to_pad(offset))
    mask1 = np.pad(mask1, xy_offset_to_pad(-offset))
    mask2 = np.pad(mask2, xy_offset_to_pad(offset))

    # Make sure that images have the same shape for sofima
    img1, img2 = homogenize_arrays_shape([img1, img2])
    mask1, mask2 = homogenize_arrays_shape([mask1, mask2])

    # Mesh represents transformation from img2 to img1. We can crop img1 to avoid OOM errors
    # There is surely a much better way to 
    pad = 200

    y1,y2,x1,x2 = mask_to_bbox(mask2)
    y1,x1 = np.array([y1,x1]) - pad
    y2,x2 = np.min([[y2 + pad, x2 + pad], img2.shape], axis=0)
    
    x = get_elastic_mesh(img1[y1:y2, x1:x2], 
                         img2[y1:y2, x1:x2], 
                         mask1[y1:y2, x1:x2], 
                         mask2[y1:y2, x1:x2], 
                         patch_size, 
                         stride,
                         **kwargs)
    
    return render_fused_slice(img1, 
                              img2, 
                              mask1, 
                              mask2,
                              x,
                              stride,
                              bbox_post=[y1,y2,x1,x2],
                              work_size=512,
                              overlap=5,
                              parallelism=parallelism,
                              post_on_top=post_on_top,
                              resize_canvas=resize_canvas)