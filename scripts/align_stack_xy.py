import os

# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     /home/autoseg/anaconda3/envs/alignment/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. 
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings('ignore', category=RuntimeWarning, message='os.fork() was called')

import json
import logging
import numpy as np
import tensorstore as ts
import sys

from pymongo import MongoClient
from tqdm import tqdm

from emprocess.utils.io import set_dataset_attributes

from emalign.align_xy.render import render_slice_xy
from emalign.align_xy.stitch_ongrid import get_coarse_offset, get_elastic_mesh
from emalign.arrays.stacks import Stack, parse_stack_info
from emalign.arrays.tile_map import get_tile_map_margins
from emalign.io.mongo import check_progress


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_stack_xy(output_path,
                   stack_name,
                   tile_maps_paths,
                   tile_maps_invert,
                   resolution,
                   offset,
                   stride,
                   apply_gaussian,
                   apply_clahe,
                   num_cores=1,
                   overwrite=False):
    
    '''Align and stitch image stack in XY. 

    Args:
        output_path (str): Path to the zarr container where the stack will be written.
        stack_name (str): Name of the stack. Will be used as the dataset name in the destination zarr container. 
        tile_maps_paths (dict): Dictionnary of slices to dictionnary of tile grid positions to paths of tifs.
        tile_maps_invert (dict): Dictionnary of tile grid positions to boolean, describing whether tiles need to be inverted at that position. 
        resolution (`list` of `int`): List of 2 int corresponding to the YX resolution in nanometers.
        offset (`list` of `int`): List of 3 int corresponding to the ZYX offset of the stack in voxels.
        stride (int): YX stride for computing the elastic mesh, in pixels. 
        prelim_overlap (int): Likely overlap between tiles. Overlap will be finely determined within a window given by this value.
        apply_gaussian (bool): Whether to apply a gaussian filter to tiles for denoising.
        apply_clahe (bool): Whether to apply CLAHE to tiles to enhance contrast.
        num_cores (int): Number of CPUs to use for rendering stitched images. Defaults to 1.
        overwrite (bool): Whether to overwrite dataset. If True, will delete existing dataset and start over. If False, will check for progress and skip processed slices. Defaults to False.
    '''

    if overwrite:
        logging.warning('Existing dataset will be deleted and aligned from scratch.')

    db_host=None
    project = os.path.basename(output_path).rstrip('.zarr')
    db_name=f'alignment_progress_{project}'
    collection_name='XY_' + stack_name

    client = MongoClient(db_host)
    db = client[db_name]
    collection_progress = db[collection_name]

    stack = Stack(stack_name=stack_name, 
                  tile_maps_paths=tile_maps_paths, 
                  tile_maps_invert=tile_maps_invert)

    # Variables
    zarr_path = os.path.join(output_path, stack.stack_name)
    zarr_path_mask = os.path.join(output_path, stack.stack_name + '_mask')
    attrs_path = os.path.join(zarr_path, '.zattrs')

    z_offset = min(stack.slices)
    z_shape  = max(stack.slices)-min(stack.slices)
    offset[0] = z_offset

    # Skip if already fully processed
    if os.path.exists(attrs_path) and not overwrite:
       logging.info(f'Skipping {stack.stack_name} because it was already processed.')
       return False
    
    if overwrite or not os.path.exists(zarr_path):
        dataset = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': zarr_path,
                                        },
                            'metadata':{
                                'shape': [z_shape + 1, 
                                            1, 1],
                                'chunks':[1,512,512]
                                        },
                            'transform': {'input_labels': ['z', 'y', 'x']}
                            },
                            dtype=ts.uint8, 
                            create=True,
                            delete_existing=True).result()   
        
        dataset_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': zarr_path_mask,
                                        },
                            'metadata':{
                                'shape': [z_shape + 1, 
                                            1, 1],
                                'chunks':[1,512,512]
                                        },
                            'transform': {'input_labels': ['z', 'y', 'x']}
                            },
                            dtype=ts.bool,
                            create=True,
                            delete_existing=True).result()   
    else:
        dataset = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': zarr_path,
                                        },
                            },
                            dtype=ts.uint8).result()  
        dataset_mask = ts.open({'driver': 'zarr',
                            'kvstore': {
                                'driver': 'file',
                                'path': zarr_path_mask,
                                        },
                            },
                            dtype=ts.bool).result()  
        
    #####################
    ### PROCESS STACK ###
    #####################
    pbar = tqdm(stack.slices, position=2, desc=f'{stack.stack_name}: Processing', leave=False)
    for z in pbar:
        if check_progress({'stack_name': stack.stack_name, 'z': z}, db_host, db_name, collection_name) and not overwrite:
            pbar.set_description(f'{stack.stack_name}: Skipping...')
            continue
        pbar.set_description(f'{stack.stack_name}: Loading tile_map...')
        tm = stack.get_tile_map(z, apply_gaussian, apply_clahe)
        tile_map = tm.tile_map
        overlap = tm.estimate_overlap(scale=0.1)
        
        if len(tile_map) > 1:
            # There are more than one tiles            
            pbar.set_description(f'{stack.stack_name}: Computing elastic meshes...')
            # Compute overlap for better coarse mesh estimation
            overlap_pad = 80
            cx, cy, coarse_mesh = get_coarse_offset(tile_map, 
                                                    tm.tile_space,
                                                    overlap=[overlap,               # try first
                                                             overlap+overlap_pad]   # try second
                                                   )

            if overlap > 160:
                # Generally good parameters
                render_stride=stride
                patch_size = 160
                k0 = 0.01
                k = 0.1
                gamma = 0.5

                # Determine margin by finding the minimum displacement in X or Y between adjacent tiles
                # Margin is how many pixels to ignore from the tiles when rendering. Too high leaves a delimitation, too low leaves a gap
                min_displacement = np.abs(np.concatenate([cx[0,0,0,:][~np.isnan(cx[0,0,0,:])], 
                                                          cy[1,0,0,:][~np.isnan(cy[1,0,0,:])]])).min()
                margin = min(200, int(min_displacement // 2 * 0.9))
            else:
                # Parameters tested for very small overlap
                render_stride=10
                patch_size=30
                k0=0.07
                k=0.2
                gamma=0.5
                margin=10
            
            meshes = get_elastic_mesh(tile_map, 
                                      cx, 
                                      cy, 
                                      coarse_mesh,
                                      stride=render_stride,
                                      patch_size=patch_size,
                                      k0=k0,
                                      k=k,
                                      gamma=gamma)
            # Ensure that first tiles acquired are rendered last because they are sharper and should be on top
            meshes = {k:meshes[k] for k in sorted(meshes)[::-1]}
            margin_map = get_tile_map_margins(tm.tile_space, margin)
                                      
            pbar.set_description(f'{stack.stack_name}: Rendering...')
            parallelism = min(num_cores, len(tile_map))
            stitch_score = render_slice_xy(dataset, z-z_offset, tile_map, meshes, render_stride, tm.tile_masks, 
                                           parallelism=parallelism, margin_overrides=margin_map, dest_mask=dataset_mask)
        else:
            # There is only one tile, no need to compute anything
            pbar.set_description(f'{stack.stack_name}: Writing unique tile...')
            stitch_score = render_slice_xy(dataset, z-z_offset, tile_map, None, None, None, parallelism=1, dest_mask=dataset_mask)

        if np.any(stitch_score == 0) or np.isnan(stitch_score).any():
            logging.warning(f'{stack.stack_name}: stitch score too low, tiles may not overlap if margin is too large (z = {z})')

        doc = {
            'stack_name': stack.stack_name,
            'z': z,
            'mesh_parameters':{
                            'stride':render_stride,
                            'patch_size':patch_size,
                            'k0':k0,
                            'k':k,
                            'gamma':gamma
                              },
            'overlap': overlap,
            'margin': margin,
            'stitch_score': float(np.median(stitch_score)),
            'missing_tile': tm.missing_tiles
              }
        collection_progress.insert_one(doc)

    pbar.set_description(f'{stack.stack_name}: done')

    # Attributes are ZYX coordinates
    # Resolution in Z is hard coded to be 50 nm currently
    # Voxel_size and resolution are both set for compatibility with different versions of funlib/daisy/gunpowder
    # Keys are used in subsequent steps in the alignment and segmentation pipeline
    attributes = {'voxel_offset': offset,
                  'offset': list(map(int, np.array(offset)*np.array([50, *resolution]))),
                  'resolution': list(map(int, (50, *resolution))),
                  'voxel_size': list(map(int, (50, *resolution)))}

    set_dataset_attributes(dataset, attributes)

    return True


if __name__ == '__main__':

    config_path = sys.argv[1]
    stack_name  = sys.argv[2]
    num_cores = int(sys.argv[3])

    with open(config_path, 'r') as f:
        main_config = json.load(f)

    main_dir        = main_config['main_dir']
    output_path     = main_config['output_path']
    resolution      = main_config['resolution']
    offset          = main_config['offset']
    stride          = main_config['stride']
    prelim_overlap  = main_config['overlap']
    apply_gaussian  = main_config['apply_gaussian']
    apply_clahe     = main_config['apply_clahe']
    stack_configs   = main_config['stack_configs']
    
    tile_maps_paths, tile_maps_invert = parse_stack_info(stack_configs[stack_name])

    align_stack_xy(output_path=output_path,
                   stack_name=stack_name,
                   tile_maps_paths=tile_maps_paths,
                   tile_maps_invert=tile_maps_invert,
                   resolution=resolution,
                   offset=offset,
                   stride=stride,
                   apply_gaussian=apply_gaussian,
                   apply_clahe=apply_clahe,
                   num_cores=num_cores,
                   overwrite=False)
