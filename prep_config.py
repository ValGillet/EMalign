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
import sys

from emalign.utils.stacks_utils import Stack
from emalign.utils.io_utils import *
from emalign.utils.align_xy_utils import *
from emalign.utils.check_utils import *


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def prep_align_stacks(main_dir,
                      config_dir,
                      output_path,
                      dir_pattern,
                      resolution,
                      offset,
                      stride,
                      overlap,
                      scale,
                      apply_gaussian,
                      apply_clahe,
                      num_workers,
                      port):
    
    logging.info(f'Configs will be stored at: {config_dir}')
    os.makedirs(config_dir, exist_ok=True)

    if os.path.exists(os.path.join(config_dir, 'main_config.json')):
        logging.info('Config already exists in dir, exiting process...')
        sys.exit()

    # Find tilesets with wanted resolution
    logging.info(f'Looking for tilesets in: {main_dir}')
    stack_paths = get_tilesets(main_dir, resolution, dir_pattern, num_workers)

    logging.info(f'Found {len(stack_paths)} directories corresponding to resolution {resolution}: ')
    for s in stack_paths:
        logging.info(f'    {s}')

    # Invert stack?
    logging.info('Please check whether to invert stacks')
    invert_instructions = check_stacks_to_invert(stack_paths, resolution, num_workers, port=port)      

    stacks = []
    for stack_path in stack_paths:
        stack = Stack(stack_path)
        stack._get_tilemaps_paths()
        for k in stack.tile_maps_invert.keys():
            stack.tile_maps_invert[k]=invert_instructions[stack.stack_name]
        stacks.append(stack)    

    # Look for overlapping stacks
    slice_to_paths = defaultdict(dict)
    for stack in stacks:
        for z in stack.slices:
            slice_to_paths[z].update({stack.stack_name: stack})
    stack_pairs = np.unique([list(z.keys()) for z in slice_to_paths.values() if len(z) > 1], axis=0)       
    print(slice_to_paths)

    logging.info(f'Found stack pairs: {stack_pairs}')
    # Overlapping stacks were found, figure out overlapping regions
    combined_stacks = []
    for pair in stack_pairs:
        stack_1, stack_2 = [stack for stack in stacks if stack.stack_name in pair]
        # Detect overlapping regions
        combined_stack = check_combined_stacks(stack_1, 
                                               stack_2, 
                                               overlap, 
                                               apply_gaussian, 
                                               apply_clahe, 
                                               scale,
                                               resolution,
                                               port)
        
        combined_stacks.append(combined_stack)

    stacks = [s for s in stacks if s.stack_name not in stack_pairs]

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
        config_path = os.path.join(config_dir, stack.stack_name + '.json')
        config_paths.update({stack.stack_name: os.path.abspath(config_path)})      

        with open(config_path, 'w') as f:
            json.dump(config_stack, f, indent='')

    for stack in combined_stacks:
        json_tile_maps = {}
        for z, tile_map in stack.slice_to_tilemap.items():
            json_tile_maps[str(z)] = {str(k):v for k,v in tile_map.items()}

        config_stack = {'combined': True,
                        'z_start': stack.slices[0],
                        'z_end': stack.slices[-1],
                        'tile_maps': json_tile_maps,
                        'tile_maps_invert': {str(k):v for k,v in stack.tile_maps_invert.items()},
                        }
        config_path = os.path.join(config_dir, stack.stack_name + '.json')
        config_paths.update({stack.stack_name: os.path.abspath(config_path)})
            
        with open(config_path, 'w') as f:
            json.dump(config_stack, f, indent='')

    main_config = {
                'main_dir': os.path.abspath(main_dir),
                'stack_configs': config_paths,
                'tilesets_combined': len(stack_pairs),
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
    parser.add_argument('-cfg', '--config_dir',
                        metavar='OUT_DIR',
                        dest='config_dir',
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
    parser.add_argument('-p', '--dir_pattern',
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
    args=parser.parse_args()


    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}')

    prep_align_stacks(**vars(args))    