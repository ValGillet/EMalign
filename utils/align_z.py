import jax
import jax.numpy as jnp
import numpy as np
import os
import tensorstore as ts

from concurrent import futures
from connectomics.common import bounding_box
from cv2 import resize, equalizeHist
from sofima import flow_field, flow_utils, map_utils, mesh, warp
from tqdm import tqdm

from emprocess.utils.transform import rotate_image

from .io import get_data_samples, get_dataset_attributes
from .arrays import compute_mask, pad_to_shape
from .offsets import *
from emalign.utils.tile_map_positions import estimate_transform_sift


def get_data(dataset, z, offset, target_scale, rotation_angle=0):
    try:
        data = dataset[z, ...].read().result()
        data = equalizeHist(data)

        if target_scale < 1:
            data = resize(data, None, fx=target_scale, fy=target_scale)

        if rotation_angle != 0:
            data = rotate_image(data, rotation_angle)

        data = np.pad(data, np.stack([offset[1:], [0,0]]).T)
        return data
    except Exception as e:
        raise RuntimeError(e)


def compute_datasets_offsets(datasets, 
                             offsets,
                             range_limit,
                             scale, 
                             filter_size,
                             step_slices,
                             yx_target_resolution,
                             pad_offset=(0,0),
                             num_workers=0):
    
    offsets_yx = [[0,0]]
    fs = []
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        for dataset in datasets:
            fs.append(tpe.submit(get_data_samples, dataset, step_slices, yx_target_resolution))

        fs = fs[::-1]

        # Do very first dataset
        data = fs.pop().result()
        inner_offsets = [estimate_rough_z_offset(data[i-1], data[i], scale=scale, range_limit=range_limit, filter_size=filter_size)[0] 
                        for i in range(1, len(data))]
        # Offset between first and last image of the first dataset
        last_inner_offset = np.sum(inner_offsets, axis=0)
        rotation_angle = 0
        for _ in tqdm(range(len(fs)),
                      desc=f'Calculating offset between {len(datasets)} datasets.'):
            # Reference is the latest image before the current dataset
            prev = rotate_image(data[-1], rotation_angle)
            data = fs.pop().result()

            _, rotation_angle, _ = estimate_transform_sift(prev, data[0], 0.3)
            # Calculate offset to the last stack 
            offset_to_last, _ = estimate_rough_z_offset(prev, rotate_image(data[0], rotation_angle), scale=scale, range_limit=range_limit, filter_size=filter_size)
            offsets_yx.append(offset_to_last + last_inner_offset)

            # Offset between first and last image (to account for differences between first images of different stacks and drift)
            last_inner_offset = np.sum([estimate_rough_z_offset(data[i-1], data[i], scale=scale, range_limit=range_limit, filter_size=filter_size)[0] 
                                    for i in range(1, len(data))], axis=0)

    offsets_yx = np.array(offsets_yx)

    yx_cumsum = np.cumsum(offsets_yx, axis=0)
    offsets[:, 1:] += (yx_cumsum - np.min(yx_cumsum, axis=0)).astype(int)
    offsets[:, 1:] += np.array(pad_offset)
    return offsets


def _compute_flow(dataset, 
                  offset, 
                  patch_size, 
                  stride, 
                  scale, 
                  filter_size, 
                  range_limit,
                  first_slice=None,
                  target_scale=1,
                  rotation_angle=0,
                  num_threads=0):

    dataset_name = dataset.kvstore.path.split('/')[-2]

    if scale < 1:
        dataset = ts.downsample(dataset, [1,
                                         int(1/scale),
                                         int(1/scale)], 'mean')

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    if first_slice is None:
        # Use dataset's first slice to compute flow from.
        start = 1
        prev = get_data(dataset, start-1, offset, target_scale)

        while not prev.any():
            start += 1
            prev = get_data(dataset, start-1, offset, target_scale)
    else:
        # Use provided first slice to compute flow from. Could be slice of a previous dataset
        start = 0
        prev = resize(first_slice, None, fx=scale, fy=scale) if scale<1 else first_slice
    
    prev_mask = compute_mask(prev, filter_size, range_limit)

    flows = []
    for z in tqdm(range(start, dataset.shape[0]),
                    position=0,
                    desc=f'{dataset_name}: Computing flow (scale={scale})'):
        curr = get_data(dataset, z, offset, target_scale, rotation_angle=rotation_angle)

        if not curr.any():
            # If empty slice, compare to the next one
            continue

        try:
            # if z == 0:
            # Different shapes may cause issues so we need to bring prev to the right shape without losing info
            # Note that we don't want to change the shape of curr if we can avoid it because then we'd have to keep track for 
            # the whole pipeline since the flow shape will have changed too.
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

            # First slice comes from a different dataset which may have black tiles
            # We use masks to ensure that only regions with data are used to compute the match
            curr_mask = compute_mask(curr, filter_size, range_limit)

            flow = mfc.flow_field(prev, curr, (patch_size, patch_size),
                                    (stride, stride), batch_size=256,
                                    pre_mask=prev_mask, post_mask=curr_mask)
            # else:
            #     # We could use masks here too, but slices within a dataset match fairly well already and computing masks takes time
            #     flow = mfc.flow_field(prev, curr, (patch_size, patch_size),
            #                             (stride, stride), batch_size=512)
            flows.append(flow)

            prev = curr
            prev_mask = curr_mask
        except Exception as e:
            raise RuntimeError(e)
    jax.clear_caches()
    return np.transpose(np.array(flows), [1, 0, 2, 3]) # [channels, z, y, x]


def compute_flow_dataset(dataset, 
                         offset, 
                         scale, 
                         patch_size, 
                         stride, 
                         max_deviation,
                         max_magnitude,
                         filter_size, 
                         range_limit,
                         first_slice=None,
                         target_scale=None,
                         rotation_angle=0,
                         num_threads=0):
    
    dataset_name = dataset.kvstore.path.split('/')[-2]

    flow = _compute_flow(dataset=dataset, 
                         offset=offset, 
                         patch_size=patch_size, 
                         stride=stride, 
                         scale=1, 
                         filter_size=filter_size, 
                         range_limit=range_limit, 
                         first_slice=first_slice, 
                         target_scale=target_scale, 
                         rotation_angle=rotation_angle, 
                         num_threads=num_threads)
    ds_flow = _compute_flow(dataset=dataset, 
                            offset=(offset*scale).astype(int), 
                            patch_size=patch_size, 
                            stride=stride, 
                            scale=scale, 
                            filter_size=filter_size, 
                            range_limit=range_limit, 
                            first_slice=first_slice, 
                            target_scale=target_scale, 
                            rotation_angle=rotation_angle, 
                            num_threads=num_threads)

    pad = patch_size // 2 // stride
    flow = np.pad(flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    ds_flow = np.pad(ds_flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    flow = flow_utils.clean_flow(flow, 
                                 min_peak_ratio=1.6, 
                                 min_peak_sharpness=1.6, 
                                 max_magnitude=max_magnitude, 
                                 max_deviation=max_deviation)
    ds_flow = flow_utils.clean_flow(ds_flow, 
                                    min_peak_ratio=1.6, 
                                    min_peak_sharpness=1.6, 
                                    max_magnitude=max_magnitude, 
                                    max_deviation=max_deviation)
    
    ds_flow_hires = np.zeros_like(flow)

    bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                    size=(flow.shape[-1], flow.shape[-2], 1))
    bbox_ds = bounding_box.BoundingBox(start=(0, 0, 0), 
                                       size=(ds_flow.shape[-1], ds_flow.shape[-2], 1))

    for z in tqdm(range(ds_flow.shape[1]),
                  desc=f'{dataset_name}: Upsampling flow map'):
        # Upsample and scale spatial components.
        resampled = map_utils.resample_map(
            ds_flow[:, z:z+1, ...],  #
            bbox_ds, bbox, 
            1 / scale, 1)
        ds_flow_hires[:, z:z + 1, ...] = resampled / scale

    return flow_utils.reconcile_flows((flow, ds_flow_hires), 
                                      max_gradient=0, max_deviation=max_deviation, min_patch_size=0)


def get_inv_map(flow, stride, dataset_name, mesh_config=None):

    if mesh_config is None:
        mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=True)

    solved = [np.zeros_like(flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])

    for z in tqdm(range(0, flow.shape[1]),
                  desc=f'{dataset_name}: Relaxing mesh'):
        prev = map_utils.compose_maps_fast(flow[:, z:z+1, ...], origin, stride,
                                           solved[-1], origin, stride)
        x, _, _ = mesh.relax_mesh(np.zeros_like(solved[0]), prev, mesh_config)
        solved.append(np.array(x))

    solved = np.concatenate(solved, axis=1)

    flow_bbox = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

    inv_map = map_utils.invert_map(solved, flow_bbox, flow_bbox, stride)

    return inv_map, flow_bbox


def align_arrays_z(prev, 
                   curr, 
                   scale, 
                   patch_size, 
                   stride, 
                   max_magnitude,
                   max_deviation,
                   k0,
                   k,
                   gamma,
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
                                (stride, stride), batch_size=256,
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
                                 max_magnitude=max_magnitude, 
                                 max_deviation=max_deviation)
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

    mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=gamma, k0=k0, k=k, stride=(stride, stride), num_iters=1000,
                                         max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                         final_cap=10, prefer_orig_order=True)
    inv_map, flow_bbox = get_inv_map(flow, stride, 'Test', mesh_config)

    data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                         size=(output_shape[-1], output_shape[-2], 1))

    aligned = warp.warp_subvolume(curr_align[None, None, ...], data_bbox, inv_map[:, 1:2, ...], 
                                  flow_bbox, stride, data_bbox, 'lanczos', parallelism=num_threads)
    
    return np.stack([prev_align, curr_align]), np.stack([prev_align, aligned.squeeze()])