import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     /home/autoseg/anaconda3/envs/alignment/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. 
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called")

import argparse
import json
import logging
import numpy as np
import pandas as pd
import sys

from .align_xy.tile_map_positions import estimate_tile_map_positions
from .align_xy.prep import find_offset_from_main_config, get_stacks, check_stacks_to_invert
from .io.volumescope import get_tilesets

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def prep_align_stacks(main_dir,
                      project_dir,
                      output_path,
                      dir_pattern,
                      resolution,
                      offset,
                      stride,
                      overlap,
                      scale,
                      apply_gaussian,
                      apply_clahe,
                      prev_cfg,
                      num_workers,
                      port):
    
    config_dir = os.path.join(project_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    logging.info(f'Configs will be stored at: {project_dir}')

    if os.path.exists(os.path.join(config_dir, 'main_config.json')):
        logging.info('Config already exists in dir, exiting process...')
        sys.exit()

    if prev_cfg is not None:
        offset[0] = find_offset_from_main_config(prev_cfg)
        logging.info(f'Determined z offset from previous dataset: {offset[0]}')

    # Find tilesets with desired resolution
    logging.info(f'Looking for tilesets in: {main_dir}')
    stack_paths = get_tilesets(main_dir, resolution, dir_pattern, num_workers)

    logging.info(f'Found {len(stack_paths)} directories corresponding to resolution {resolution}: ')
    for s in stack_paths:
        logging.info(f'    {s}')
        
    if not stack_paths:
        sys.exit(f'No directory corresponding to the query was found at {main_dir}')

    # Invert stack?
    logging.info('Please check whether to invert stacks')
    invert_instructions = check_stacks_to_invert(stack_paths, resolution, num_workers, port=port)

    stacks = get_stacks(stack_paths, invert_instructions)

    # Look for overlapping stacks
    combined_stacks = {k:v for k,v in stacks.items() if isinstance(v, list)}
    stacks = [v for v in stacks.values() if not isinstance(v, list)]

    logging.info(f'Found {len(combined_stacks)} combined stack')
    processed_combined_stacks = []
    if len(combined_stacks) > 0:
        logging.info('Checking groups of overlapping stacks')
        # Overlapping stacks were found, figure out overlapping regions
        for combined_stack in combined_stacks.values():
            # Detect overlapping regions
            processed_combined_stacks += estimate_tile_map_positions(combined_stack, 
                                                                     apply_gaussian, 
                                                                     apply_clahe, 
                                                                     scale=[0.3, 0.5], 
                                                                     overlap_score_threshold=0.8,
                                                                     rotation_threshold=5)

    logging.info('Writing stack configs')
    config_paths = {}
    for stack in stacks:
        json_tile_maps = {}
        for z, tile_map in stack.slice_to_tilemap.items():
            json_tile_maps[str(z)] = {str(k):v for k,v in tile_map.items()}

        config_stack = {'combined': False,
                        'z_start': stack.slices[0],
                        'z_end': stack.slices[-1],
                        'tile_maps': json_tile_maps,
                        'tile_maps_invert': {str(k):v for k,v in stack.tile_maps_invert.items()},
                        }
        config_path = os.path.join(config_dir, 'xy_' + stack.stack_name + '.json')
        config_paths.update({stack.stack_name: os.path.abspath(config_path)})      

        with open(config_path, 'w') as f:
            json.dump(config_stack, f, indent='')

    for stack in processed_combined_stacks:
        json_tile_maps = {}
        for z, tile_map in stack.slice_to_tilemap.items():
            json_tile_maps[str(z)] = {str(k):v for k,v in tile_map.items()}

        config_stack = {'combined': True,
                        'z_start': stack.slices[0],
                        'z_end': stack.slices[-1],
                        'tile_maps': json_tile_maps,
                        'tile_maps_invert': {str(k):v for k,v in stack.tile_maps_invert.items()},
                        }
        config_path = os.path.join(config_dir, 'xy_' + stack.stack_name + '.json')
        config_paths.update({stack.stack_name: os.path.abspath(config_path)})
            
        with open(config_path, 'w') as f:
            json.dump(config_stack, f, indent='')

    project_name = input('Please name the project: ')

    main_config = {
                'project_name': project_name,
                'main_dir': os.path.abspath(main_dir),
                'stack_configs': config_paths,
                'tilesets_combined': len(combined_stacks),
                'resolution': resolution,
                'offset': offset,
                'output_path': os.path.abspath(output_path),
                'scale': scale,
                'stride': stride,
                'overlap': overlap,
                'apply_gaussian': apply_gaussian,
                'apply_clahe': apply_clahe
                }

    with open(os.path.join(config_dir, 'main_config.json'), 'w') as f:
            json.dump(main_config, f, indent='')


if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-i', '--input_dir',
                        metavar='MAIN_DIR',
                        dest='main_dir',
                        required=True,
                        type=str,
                        help='Path to the directory containing tilesets. \n \
                              This directory contains subdirectories themselves containing the tiles to align. \n \
                              Subdirectories are expected to contain tif and info files.')
    parser.add_argument('-o', '--output_zarr',
                        metavar='OUT_ZARR',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='Path to the zarr container where to write stitched tifs.')
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Directory where the config will be written.')
    parser.add_argument('-r', '--resolution',
                        metavar='RESOLUTION',
                        dest='resolution',
                        required=True,
                        type=int,
                        nargs=2,
                        default=None,
                        help='XY resolution to align. Will look into the info file of each directory to find the tileset with the wanted resolution.')
    parser.add_argument('--offset',
                        metavar='OFFSET',
                        dest='offset',
                        required=False,
                        type=int,
                        nargs=3,
                        default=[0,0,0],
                        help='ZYX pixel offset for the final volume. Default: [0,0,0]')
    parser.add_argument('--overlap',
                        metavar='OVERLAP',
                        dest='overlap',
                        type=int,
                        default=500,
                        help='Size of the overlapping region in pixels, or in size ratio. Default: 500')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=None,
                        help='Number of cores to use for multiprocessing and multithreading. Default: 0 (all cores available)')
    parser.add_argument('--stride',
                        metavar='STRIDE',
                        dest='stride',
                        type=int,
                        default=20,
                        help='Stride used to compute the elastic mesh. Default: 20')    
    parser.add_argument('--scale',
                        metavar='SCALE',
                        dest='scale',
                        type=int,
                        default=0.5,
                        help='Downsampling scale to use for computing the elastic mesh. Lower values speed up the process but may fail. \
                              If a stack fails, it will temporarily use scale=1 (no downsampling). \
                              Between 0 and 1. Default: 0.5')
    parser.add_argument('--not-apply_gaussian',
                        dest='apply_gaussian',
                        action='store_false',
                        default=True,
                        help='Don\'t apply a gaussian filter to images before alignment.') 
    parser.add_argument('--not-apply_clahe',
                        dest='apply_clahe',
                        action='store_false',
                        default=True,
                        help='Don\'t apply CLAHE to images before alignment.') 
    parser.add_argument('--dir-pattern',
                        metavar='DIR_PATTERN',
                        dest='dir_pattern',
                        nargs=1,
                        default=[''],
                        type=str,
                        help='Pattern to match for subdirectories to process. Default to no pattern')
    parser.add_argument('--port',
                        metavar='PORT',
                        dest='port',
                        type=int,
                        default=33333,
                        help='Port used by neuroglancer')
    parser.add_argument('--prev-cfg',
                        metavar='PREV_CFG',
                        dest='prev_cfg',
                        default=None,
                        type=str,
                        help='Path to the main_config of a previous part of the dataset. If provided, the z offset will be determined from the previous dataset.')
    
    args=parser.parse_args()


    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}')

    prep_align_stacks(**vars(args)) 