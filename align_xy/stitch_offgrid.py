'''
Functions for stitching images existing on the same XY plane.
Contrary to tilemaps, these functions are for images that are not on a predictable grid. 
'''

import numpy as np

from connectomics.common import bounding_box
from scipy.ndimage import binary_erosion
from sofima import flow_field, flow_utils, mesh
from sofima.warp import ndimage_warp

from emprocess.utils.mask import compute_greyscale_mask
from emprocess.utils.transform import rotate_image

from ..array.sift import estimate_transform_sift
from ..array.pad import xy_offset_to_pad
from ..array.utils import homogenize_arrays_shape


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
    x, _, _ = mesh.relax_mesh(np.zeros_like(flow), -flow, mesh_config)
    return np.array(x)[:,0,...]


def render_fused_slice(pre, 
                       post,
                       pre_mask,
                       post_mask,
                       x,
                       stride,
                       work_size=512,
                       overlap=1,
                       parallelism=1,
                       output_shape=None,
                       return_warped=False
                       ):
    
    # By default, output shape should be the same as any one of the images as they should have been roughly aligned
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

    if return_warped:
        return warped_img, warped_mask

    # Stitch images together. Erode mask slightly to have a nicer seam
    eroded_mask = binary_erosion(warped_mask, iterations=2)
    stitched = pre.copy()
    stitched[eroded_mask.astype(bool)] = warped_img[eroded_mask.astype(bool)]
    return stitched, pre_mask | eroded_mask


def stitch_images(img1, 
                img2,
                mask1=None, 
                mask2=None,
                scale=0.1,
                patch_size=160,
                stride=40,
                parallelism=1,
                **kwargs):
    
    '''
    Stitch two images on the same slice. Img1 is the reference, img2 is the moving image.
    '''

    # Compute mask if not provided
    mask1 = mask1 if mask1 is not None else compute_greyscale_mask(img1)
    mask2 = mask2 if mask2 is not None else compute_greyscale_mask(img2)

    # Estimate rigid offset
    offset, rotation, _ = estimate_transform_sift(img1, img2, scale)

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

    print(img1.shape, img2.shape)

    x = get_elastic_mesh(img1, 
                         img2, 
                         mask1, 
                         mask2, 
                         patch_size, 
                         stride,
                         **kwargs)
    
    return render_fused_slice(img1, 
                              img2, 
                              mask1, 
                              mask2,
                              x,
                              stride,
                              work_size=512,
                              overlap=1,
                              parallelism=parallelism)