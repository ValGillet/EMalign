import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '1'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import json
import numpy as np
import logging
import sys
import tensorstore as ts

from concurrent import futures
from connectomics.common import bounding_box
from cv2 import resize
from tqdm import tqdm

from sofima import warp, mesh
from emalign.utils.offsets import estimate_rough_z_offset
from emalign.utils.align_z import compute_flow_dataset, get_inv_map, get_data
from emalign.utils.io import get_dataset_attributes, set_dataset_attributes
from emalign.utils.tile_map_positions import estimate_transform_sift

from emprocess.utils.transform import rotate_image

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# TODO: save PR metric for optic flow to highlight slices where alignment might not be good

def align_stack_z(destination_path,
                  dataset_path, 
                  offset, 
                  scale, 
                  patch_size, 
                  stride, 
                  max_deviation,
                  max_magnitude,
                  k0, 
                  k, 
                  gamma,
                  filter_size,
                  range_limit,
                  first_slice,
                  yx_target_resolution,
                  num_threads,
                  save_downsampled=1,
                  overwrite=False):
    
    dataset_name = dataset_path.split('/')[-2]
    destination_path = os.path.abspath(destination_path)
    dataset_path = os.path.abspath(dataset_path)
    
    # Open input dataset
    dataset = ts.open({'driver': 'zarr',
                       'kvstore': {
                             'driver': 'file',
                             'path': dataset_path,
                                  }
                      },
                      dtype=ts.uint8
                      ).result()
    
    # Check the attributes file for a variable that would show that this stack was processed
    attrs = get_dataset_attributes(dataset)
    
    if attrs.get('z_aligned', False) == True and not overwrite:
        logging.info(f'Dataset {dataset_name} was already processed and will be skipped.')
        return False

    # Make the resolution match the target
    res = attrs['resolution'][-1]
    if yx_target_resolution is None:
        target_scale = 1
    else:
        target_scale = res/yx_target_resolution[0]

        assert yx_target_resolution[0] == yx_target_resolution[1], 'Target resolution must be the same for X and Y.'
        assert target_scale <= 1, 'Target resolution must be lower than current dataset resolution.'
    logging.info(f'Target scale: {target_scale}')
    
    offset = np.array(offset)

    # Open destination
    destination = ts.open({'driver': 'zarr',
                           'kvstore': {
                                 'driver': 'file',
                                 'path': destination_path,
                                      }
                          },
                          dtype=ts.uint8
                          ).result()
    
    if save_downsampled > 1:
        ds_output_path, project_name = destination_path.rsplit('/', maxsplit=1)
        ds_output_path = os.path.join(ds_output_path, f'{save_downsampled}x_' + project_name)
        ds_destination = ts.open({'driver': 'zarr',
                                  'kvstore': {
                                          'driver': 'file',
                                          'path': ds_output_path,
                                              }
                                  },
                                  dtype=ts.uint8
                                  ).result()
        
    # Get first slice
    if first_slice is not None:
        i = int(first_slice)
        first_slice = destination[first_slice].read().result()
        while not first_slice.any():
            # If latest slice before this dataset is empty, go to the previous one until finding a non-empty slice
            i -= 1
            first_slice = destination[i].read().result()

        # Re-compute offset to account for drift during z align of the previous stack(s)
        # Get slice without offset because we want to apply a rotation before applying the offset
        # Based on intuition, I have not tested whether that made a difference
        test_slice = get_data(dataset, 0, [0,0,0], target_scale, rotation_angle=0)
        sift_offset, rotation_angle, _ = estimate_transform_sift(first_slice, test_slice, 0.3)
        print(-sift_offset)
        print(rotation_angle)
        test_slice = rotate_image(test_slice, rotation_angle)
        sift_offset, _, _ = estimate_transform_sift(first_slice, test_slice, 0.3)
        sift_offset = -sift_offset.astype(int)[::-1]
        print(sift_offset)
        # yx_offset = estimate_rough_z_offset(first_slice, test_slice, scale=0.1, range_limit=10)[0]     
        # if np.isnan(yx_offset).any():
        #     yx_offset = -sift_offset.astype(int)[::-1]

        assert not np.isnan(sift_offset).any(), f'Images might not overlap enough. Is there too much padding following a rotation? ({rotation_angle})'
        assert np.all(sift_offset > 0), f'Offset should be positive ({sift_offset}). Did the images drift over time?'

        offset[1:] = sift_offset
    else:
        rotation_angle=0
        
    # Compute flow
    flow = compute_flow_dataset(dataset, 
                                offset, 
                                scale, 
                                patch_size, 
                                stride, 
                                max_deviation,
                                max_magnitude,
                                filter_size,
                                range_limit,
                                first_slice,
                                target_scale,
                                rotation_angle,
                                num_threads)
    
    # Elasticity ratio = k0/k (the larger the more deformation)
    # The ratio is what matters for how much the data is deformed
    # However smaller numbers might limit the possible deviation of the mesh
    # Good values:
    # k0 = 0.01 # inter-section springs (elasticity). High k0 results in images that tend to "fold" onto themselves
    # k = 0.4 # intra-section springs (elasticity). Increase if data deforms too much
    # gamma = 0.5 # dampening factor. Increase if data drift over time

    mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=gamma, k0=k0, k=k, stride=(stride, stride), num_iters=1000,
                                         max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                         final_cap=10, prefer_orig_order=False)
    inv_map, flow_bbox = get_inv_map(flow, stride, dataset_name, mesh_config)

    # Get first slice
    if first_slice is None:
        # First slice to write is the first slice of the dataset, untouched but padded
        # Then we warp the rest from the next slice
        start = 1
        first = get_data(dataset, start-1, offset, target_scale, rotation_angle=rotation_angle)

        while not first.any():
            # If the first slice is empty, go to the next one in line
            # Repeat until finding a non-empty slice
            start += 1
            first = get_data(dataset, start-1, offset, target_scalerotation_angle=rotation_angle)

        y,x = first.shape
        z = offset[0]

        if np.any(destination.domain.exclusive_max < np.array([z+1, y, x])):
            # Resize the destination dataset if the new slice is larger
            new_max = np.max([destination.domain.exclusive_max, [z+1, y, x]], axis=0)
            destination = destination.resize(exclusive_max=new_max, expand_only=True).result()

        destination[z, :y, :x].write(first).result()
    else:
        # All slices have to be warped to match the last slice of the previous stack
        start = 0

    # Start alignment
    skipped = 0
    for z in tqdm(range(start, dataset.shape[0]), 
                    position=0,
                    desc=f'{dataset_name}: Rendering aligned slices'):
        # data = fs_read.pop().result()
        data = get_data(dataset, z, offset, target_scale, rotation_angle=rotation_angle)
        data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                             size=(data.shape[-1], data.shape[-2], 1))

        if not data.any():
            # If empty slice, skip and go to next z
            skipped += 1
            continue

        if first_slice is None:
            inv_z = z - skipped
        else:
            inv_z = z + 1 - skipped

        # Warp data in parallel. Use num_threads minus the 4 threads used for reading and writing
        # warp_subvolume is the bottleneck here, so 4 threads for read and write is most likely enough to keep up
        aligned = warp.warp_subvolume(data[None, None, ...], data_bbox, inv_map[:, inv_z:inv_z+1, ...], 
                                        flow_bbox, stride, data_bbox, 'lanczos', parallelism=num_threads)
        aligned = aligned[0,0,...]

        write_data(destination, aligned, z + offset[0], save_downsampled, ds_destination)
    logging.info(f'{dataset_name}: Done. ({skipped} empty slices)')

    # Add an attribute to keep track of what datasets have been aligned already
    attrs['z_aligned'] = True

    set_dataset_attributes(dataset, attrs)

    return True

def write_data(destination, data, z, save_downsampled=1, ds_destination=None):
    tasks = []

    # Write to destination
    y,x = data.shape
    if np.any(destination.domain.exclusive_max < np.array([z+1, y, x])):
        new_max = np.max([destination.domain.exclusive_max, [z+1, y, x]], axis=0)
        destination = destination.resize(exclusive_max=new_max, expand_only=True).result()
    tasks.append(destination[z, :y, :x].write(data).result())

    # Write downsampled data for inspection
    if save_downsampled > 1 and ds_destination is not None:
        ds_data = resize(data, None, fx=1/save_downsampled, fy=1/save_downsampled)
        y,x = ds_data.shape

        if np.any(ds_destination.domain.exclusive_max < np.array([z+1, y, x])):
            new_max = np.max([ds_destination.domain.exclusive_max, [z+1, y, x]], axis=0)
            ds_destination = ds_destination.resize(exclusive_max=new_max, expand_only=True).result()

        tasks.append(ds_destination[z, :y, :x].write(ds_data).result())
    
    return tasks

if __name__ == '__main__':

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    logging.info(f'patch_size = {config['patch_size']}')
    logging.info(f'stride = {config['stride']}')

    align_stack_z(**config)
