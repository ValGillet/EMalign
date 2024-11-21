import os
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
from new_utils.align_z_utils import compute_flow_dataset, get_inv_map

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
                  num_threads):
    
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
    dataset_name = dataset_path.split('/')[-2]

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
            first_slice -= 1
            first_slice = destination[first_slice].read().result()

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
        start = 1
        first = dataset[start-1, ...].read().result()

        while not first.any():
            # If empty slice, go to the next one in line
            # Repeat until finding a slice with data
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
    with futures.ThreadPoolExecutor(num_threads) as tpe:
        # Prefetch the next sections to memory so that we don't have to wait for them
        # to load when the GPU becomes available.
        # Each slice is padded to account for the offset computed upstream
        # This ensures that the slices are roughly aligned for the finer alignment
        for z in range(start, dataset.shape[0]):
            fs_read.append(tpe.submit(
                lambda z=z: np.pad(dataset[z, ...].read().result(), np.stack([offset[1:], [0,0]]).T)[None, None, ...]
                                     ))
        
        # Write to file asynchronously
        fs_read = fs_read[::-1]
        skipped = 0
        for z in tqdm(range(start, dataset.shape[0]), 
                      position=2,
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

            aligned = warp.warp_subvolume(data, data_bbox, inv_map[:, inv_z:inv_z+1, ...], 
                                          flow_bbox, stride, data_bbox, 'lanczos', parallelism=1)
            aligned = aligned[0,0,...]

            fs_write.append(tpe.submit(write_data, destination, 
                                                   aligned, 
                                                   z + offset[0]))

    # Wait for processes to finish
    for f in tqdm(futures.as_completed(fs_write), 
                  total=len(fs_write), 
                  position=2, 
                  desc=f'{dataset_name}: Writing aligned slices'):
        f.result()
    logging.info(f'{dataset_name}: Done. ({skipped} empty slices)')

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

    align_stack_z(**config)