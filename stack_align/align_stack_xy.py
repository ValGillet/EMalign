import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     /home/autoseg/anaconda3/envs/alignment/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. 
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called")

import json
import logging
import numpy as np
import tensorstore as ts

from concurrent import futures
from tqdm import tqdm

from ..utils.stacks import Stack
from ..utils.io import *
from ..utils.align_xy import *
from ..utils.inspect import *


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_stack_xy(output_path,
                   stack_name,
                   tile_maps_paths,
                   tile_maps_invert,
                   resolution,
                   stride,
                   overlap,
                   scale,
                   apply_gaussian,
                   apply_clahe,
                   num_threads):
    
    stack = Stack(stack_name=stack_name, 
                  tile_maps_paths=tile_maps_paths, 
                  tile_maps_invert=tile_maps_invert)

    # Variables
    zarr_path  = os.path.join(output_path, stack.stack_name)
    attrs_path = os.path.join(zarr_path, '.zattrs')

    z_offset = min(stack.slices)
    z_max    = max(stack.slices)-z_offset

    # Skip if already fully processed
    if os.path.exists(attrs_path):
        logging.info(f'Skipping {stack.stack_name} because it was already processed.')
        return False

    dataset = ts.open({'driver': 'zarr',
                        'kvstore': {
                            'driver': 'file',
                            'path': zarr_path,
                                    },
                        'metadata':{
                            'shape': [z_max + 1, 
                                        1, 1],
                            'chunks':[1,512,512]
                                    },
                        'transform': {'input_labels': ['z', 'y', 'x']}
                        },
                        dtype=ts.uint8, 
                        create=True,
                        delete_existing=True).result()

    #####################
    ### PROCESS STACK ###
    #####################
    fs_read = []
    fs_warp = []
    with futures.ThreadPoolExecutor(num_threads) as tpe:
        # Check if scale works for the first slice
        z = stack.slices[0]
        z, tile_map, tile_map_ds = load_tilemap({z: stack.slice_to_tilemap[z]}, 
                                                stack.tile_maps_invert,
                                                apply_gaussian, 
                                                apply_clahe,
                                                scale)  
        if scale<1:
            cx, cy, coarse_mesh = get_coarse_offset(tile_map_ds, 
                                                    overlap=overlap)
        else:
            cx, cy, coarse_mesh = get_coarse_offset(tile_map, 
                                                    overlap=overlap)

        try:
            if np.isinf(np.concatenate([cx,cy])).any():
                # If scale doesn't work, coarse offset vectors are infinite. 
                # Using non-downsampled images may solve it.
                # logging.info(f'Using scale=1 for stack: {stack.stack_name}')
                scale = 1

                cx, cy, coarse_mesh = get_coarse_offset(tile_map, 
                                                        overlap=overlap)
                
                meshes = get_elastic_mesh(tile_map, 
                                            cx, 
                                            cy, 
                                            coarse_mesh)
            else:
                meshes = get_elastic_mesh(tile_map_ds, 
                                            cx, 
                                            cy, 
                                            coarse_mesh)
                meshes = {k:rescale_mesh(v, int(1/scale)) for k,v in meshes.items()}
        except Exception as e:
            raise RuntimeError(e)
        # Queue the warping task and move on to the next slices
        fs_warp.append(tpe.submit(render_slice_xy, dataset, z-z_offset, tile_map, meshes, stride))

        # Carry on with the rest of the tasks
        for z in stack.slices[1:]:
            fs_read.append(tpe.submit(load_tilemap, 
                                        {z: stack.slice_to_tilemap[z]}, 
                                        stack.tile_maps_invert,
                                        apply_gaussian, 
                                        apply_clahe,
                                        scale))

        # Process tilemaps synchronously because functions use JAX and GPU (have not tested though)
        for f in tqdm(futures.as_completed(fs_read), total=len(fs_read), 
                      position=2, desc=f'{stack.stack_name}: Computing elastic meshes', leave=False):
            z, tile_map, tile_map_ds = f.result()
            tile_map_ds = tile_map if scale == 1 else tile_map_ds
            
            cx, cy, coarse_mesh = get_coarse_offset(tile_map_ds, 
                                                    overlap=overlap)                    
            
            meshes = get_elastic_mesh(tile_map_ds, 
                                        cx, 
                                        cy, 
                                        coarse_mesh)

            if scale < 1:
                meshes = {k:rescale_mesh(v, int(1/scale)) for k,v in meshes.items()}

            # ProcessPoolExecutor is faster for CPU intensive tasks
            fs_warp.append(tpe.submit(render_slice_xy, dataset, z-z_offset, tile_map, meshes, stride))

        # Wait for processes to finish
        pbar = tqdm(futures.as_completed(fs_warp), 
                    total=len(fs_warp), 
                    position=2, 
                    desc=f'{stack.stack_name}: Stitching and writing tiles', 
                    leave=True)
        for f in pbar:
            f.result()
        pbar.set_description(f'{stack.stack_name}: done')

    # Attributes are ZYX coordinates
    # Resolution in Z is hard coded to be 50 nm currently
    # Keys are used in subsequent steps in the alignment and segmentation pipeline
    attributes = {'voxel_offset': (z_offset, 0, 0),
                  'offset': (z_offset*50, 0, 0),
                  'resolution': (50, *resolution)}

    set_dataset_attributes(dataset, attributes)

    return True
