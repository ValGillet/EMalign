import os
from re import I
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import json
import numpy as np
import logging
import sys
import tensorstore as ts

from concurrent import futures
from connectomics.common import bounding_box
from tqdm import tqdm

from sofima import warp
from emalign.utils.align_z_utils import compute_flow_dataset, get_inv_map, estimate_rough_offset

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_stack_z(destination_path,
                  dataset_path, 
                  offset, 
                  scale, 
                  patch_size, 
                  stride, 
                  filter_size,
                  range_limit,
                  first_slice,
                  num_threads,
                  overwrite=False):
    
    dataset_name = dataset_path.split('/')[-2]

    # Check the attributes file for a variable that would show that this stack was processed
    attributes_file = os.path.join(dataset_path, '.zattrs')
    with open(attributes_file, 'r+') as f:
        attrs = json.load(f)
    if attrs.get('z_aligned', False) == True and not overwrite:
        logging.info(f'Dataset {dataset_name} was already process and will be skipped.')
        return False
    
    offset = np.array(offset)
    
    # Open input dataset
    dataset = ts.open({'driver': 'zarr',
                       'kvstore': {
                             'driver': 'file',
                             'path': dataset_path,
                                  }
                      },
                      dtype=ts.uint8
                      ).result()
    

    # Open destination
    destination = ts.open({'driver': 'zarr',
                           'kvstore': {
                                 'driver': 'file',
                                 'path': destination_path,
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
        # Commented out because resulted in a negative offset. It might not be necessary
        # yx_offset = estimate_rough_offset(first_slice, dataset[0].read().result())
        # offset[1:] = yx_offset
        
    # Compute flow
    flow = compute_flow_dataset(dataset, 
                                offset, 
                                scale, 
                                patch_size, 
                                stride, 
                                filter_size,
                                range_limit,
                                first_slice,
                                num_threads)

    inv_map, flow_bbox = get_inv_map(flow, stride, dataset_name)

    output_shape = np.array(dataset.shape[1:]) + np.array(offset[1:])
    data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                         size=(output_shape[-1], output_shape[-2], 1))

    if first_slice is None:
        # First slice to write is the first slice of the dataset, untouched but padded
        # Then we warp the rest from the next slice
        start = 1
        first = dataset[start-1, ...].read().result()

        while not first.any():
            # If the first slice is empty, go to the next one in line
            # Repeat until finding a non-empty slice
            start += 1
            first = dataset[start-1, ...].read()

        y,x = output_shape
        z = offset[0]

        if np.any(destination.domain.exclusive_max < np.array([z+1, y, x])):
            # Resize the destination dataset if the new slice is larger
            new_max = np.max([destination.domain.exclusive_max, [z+1, y, x]], axis=0)
            destination = destination.resize(exclusive_max=new_max, expand_only=True).result()

        first = np.pad(first, np.stack([offset[1:], [0,0]]).T)
        y,x = first.shape
        destination[z, :y, :x].write(first).result()
    else:
        # All slices have to be warped to match the last slice of the previous stack
        start = 0

    fs_read = []
    fs_write = []
    with futures.ThreadPoolExecutor(4) as tpe:
        # Prefetch the next sections to memory so that we don't have to wait for them
        # to load when the GPU becomes available.
        # Each slice is padded to account for the offset computed upstream
        # This ensures that the slices are roughly aligned for the finer alignment
        for z in range(start, dataset.shape[0]):
            # Data must be padded to be aligned with the flow map, and be [channels, z, y, x]]
            fs_read.append(tpe.submit(
                lambda z=z: np.pad(dataset[z, ...].read().result(), np.stack([offset[1:], [0,0]]).T)[None, None, ...]
                                     ))
        
        # Write to file asynchronously
        fs_read = fs_read[::-1]
        skipped = 0
        for z in tqdm(range(start, dataset.shape[0]), 
                      position=0,
                      desc=f'{dataset_name}: Rendering aligned slices'):
            data = fs_read.pop().result()

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
            aligned = warp.warp_subvolume(data, data_bbox, inv_map[:, inv_z:inv_z+1, ...], 
                                          flow_bbox, stride, data_bbox, 'lanczos', parallelism=num_threads-4)
            aligned = aligned[0,0,...]

            fs_write.append(tpe.submit(write_data, destination, 
                                                   aligned, 
                                                   z + offset[0]))

    # Wait for processes to finish
    for f in tqdm(futures.as_completed(fs_write), 
                  total=len(fs_write), 
                  position=0, 
                  desc=f'{dataset_name}: Writing aligned slices'):
        f.result()
    logging.info(f'{dataset_name}: Done. ({skipped} empty slices)')

    # Add an attribute to keep track of what datasets have been aligned already
    attrs['z_aligned'] = True

    with open(attributes_file, 'w') as f:
        json.dump(attrs, f, indent='')

    return True

def write_data(destination, data, z):
    y,x = data.shape
    if np.any(destination.domain.exclusive_max < np.array([z+1, y, x])):
        new_max = np.max([destination.domain.exclusive_max, [z+1, y, x]], axis=0)
        destination = destination.resize(exclusive_max=new_max, expand_only=True).result()

    return destination[z, :y, :x].write(data).result()

if __name__ == '__main__':

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

        
    logging.info(f'patch_size = {config['patch_size']}')
    logging.info(f'stride = {config['stride']}')

    align_stack_z(**config)
