import os
import logging
import numpy as np

from cv2 import resize

from connectomics.common import bounding_box
from sofima import flow_field, flow_utils, map_utils, warp

from emalign.utils.align_z import compute_mask, pad_to_shape, get_inv_map


def align_arrays_z(prev, 
                   curr, 
                   patch_size, 
                   stride, 
                   scale, 
                   filter_size, 
                   range_limit,
                   num_threads=0):
    
    output_shape = curr.shape
 
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    data = [[prev, curr], 
            [resize(prev, None, fy=scale, fx=scale),
             resize(curr, None, fy=scale, fx=scale)]]

    flows = []
    for i, (prev, curr) in enumerate(data):
        prev_mask = compute_mask(prev, filter_size, range_limit)
        curr_mask = compute_mask(curr, filter_size, range_limit)

        # Make shapes match
        if np.any(np.array(curr.shape) > np.array(prev.shape)):
            # If prev is smaller, we pad to shape with zeros to the end of the data
            # It doesn't affect offset
            prev = pad_to_shape(prev, curr.shape)
            prev_mask = pad_to_shape(prev_mask, curr.shape)
        if np.any(np.array(prev.shape) > np.array(curr.shape)):
            # If prev is larger, we crop to shape
            # Prev and curr should be roughly overlapping, so we should not be losing relevant info
            y,x = curr.shape
            prev = prev[:y, :x]
            prev_mask = prev_mask[:y, :x]
        
        if i == 0:
            # Store the arrays for output
            prev_align = prev
            curr_align = curr
        
        try:
            flow = mfc.flow_field(prev, curr, (patch_size, patch_size),
                                (stride, stride), batch_size=512,
                                pre_mask=prev_mask, post_mask=curr_mask)
        except Exception as e:
            raise RuntimeError(e)
        
        flows.append(np.transpose(flow[None, ...], [1, 0, 2, 3])) # [channels, z, y, x]

    flow, ds_flow = flows

    pad = patch_size // 2 // stride
    flow = np.pad(flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    ds_flow = np.pad(ds_flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    flow = flow_utils.clean_flow(flow, 
                                 min_peak_ratio=1.6, 
                                 min_peak_sharpness=1.6, 
                                 max_magnitude=0, 
                                 max_deviation=0)
    ds_flow = flow_utils.clean_flow(ds_flow, 
                                    min_peak_ratio=1.6, 
                                    min_peak_sharpness=1.6, 
                                    max_magnitude=0, 
                                    max_deviation=0)
    
    ds_flow_hires = np.zeros_like(flow)

    bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                    size=(flow.shape[-1], flow.shape[-2], 1))
    bbox_ds = bounding_box.BoundingBox(start=(0, 0, 0), 
                                       size=(ds_flow.shape[-1], ds_flow.shape[-2], 1))

    # Upsample and scale spatial components.
    resampled = map_utils.resample_map(
                    ds_flow[:, 0:1, ...], 
                    bbox_ds, bbox, 2, 1)
    ds_flow_hires[:, 0:1, ...] = resampled / scale

    flow = flow_utils.reconcile_flows((flow, ds_flow_hires), 
                                       max_gradient=0, max_deviation=0, min_patch_size=400)

    inv_map, flow_bbox = get_inv_map(flow, stride, 'Test')

    data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                         size=(output_shape[-1], output_shape[-2], 1))

    aligned = warp.warp_subvolume(curr_align[None, None, ...], data_bbox, inv_map[:, 1:2, ...], 
                                  flow_bbox, stride, data_bbox, 'lanczos', parallelism=num_threads)
    
    return np.stack([prev_align, curr_align]), np.stack([prev_align, aligned.squeeze()])