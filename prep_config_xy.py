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

from emalign.utils.stacks import Stack
from emalign.utils.io import *
from emalign.utils.align_xy import *
from emalign.utils.inspect import *
from emalign.utils.tile_map_positions import estimate_tile_map_positions

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def get_stacks(stack_paths, invert_instructions):

    # Load stacks
    stacks = []
    for stack_path in stack_paths:
        stack = Stack(stack_path)
        stack._get_tilemaps_paths()
        for k in stack.tile_maps_invert.keys():
            stack.tile_maps_invert[k]=invert_instructions[stack.stack_name]
        stacks.append(stack) 

    # Split stacks if there are overlaps
    unique_slices = sorted(np.unique(np.concatenate([stack.slices for stack in stacks])).tolist())
    df = pd.DataFrame({'z': unique_slices, 
                    'stack_name': [[] for _ in range(len(unique_slices))], 
                    'tile_paths':[[] for _ in range(len(unique_slices))]
                    })

    for stack in stacks:
        for z in stack.slices:
            # Join existing name and this stack at that slice
            df.loc[df.z == z, ['stack_name']] += [[stack.stack_name]]

            # Concatenate tile paths
            df.loc[df.z == z, ['tile_paths']] += [[stack.slice_to_tilemap[z]]]

    df['group'] = df['stack_name'].ne(df['stack_name'].shift()).cumsum()

    new_stacks = {}
    for group, group_df in df.groupby('group'):    
        stack_names = group_df.stack_name.iloc[0]

        if len(stack_names) == 1:
            # Stack name becomes name + group (gives an idea of order too)
            new_stack_name = str(group).zfill(2) + '_' + stack_names[0]

            tile_map = {}
            for z in group_df.z:
                tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[0]
            
            stack = Stack()
            stack.stack_name = new_stack_name
            stack._set_tilemaps_paths(tile_map)
            stack.tile_maps_invert = {k: invert_instructions[stack_names[0]] for k in tile_map[z].keys()}

            new_stacks[new_stack_name] = stack
        
        else:
            combined_stack_name = '_'.join([str(group).zfill(2)] + stack_names)
            pair = []
            for i in range(len(stack_names)):
                new_stack_name = str(group).zfill(2) + '_' + stack_names[i]
                
                tile_map = {}
                for z in group_df.z:
                    tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[i]

                stack = Stack()
                stack.stack_name = new_stack_name
                stack._set_tilemaps_paths(tile_map)
                stack.tile_maps_invert = {k: invert_instructions[stack_names[i]] for k in tile_map[z].keys()}

                pair.append(stack)
            new_stacks[combined_stack_name] = pair
        
    return new_stacks


def find_offset_from_previous(main_config_path):
    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    z_offsets = []
    for stack_config in main_config['stack_configs'].values():
        with open(stack_config, 'r') as f:
            stack_config = json.load(f)
        
        z_offsets.append(stack_config['z_end'])

    return max(z_offsets) + 1


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
        offset[0] = find_offset_from_previous(prev_cfg)
        logging.info(f'Determined z offset from previous dataset: {offset[0]}')

    # Find tilesets with desired resolution
    logging.info(f'Looking for tilesets in: {main_dir}')
    stack_paths = get_tilesets(main_dir, resolution, dir_pattern, num_workers)

    logging.info(f'Found {len(stack_paths)} directories corresponding to resolution {resolution}: ')
    for s in stack_paths:
        logging.info(f'    {s}')

    # Invert stack?
    logging.info('Please check whether to invert stacks')
    invert_instructions = check_stacks_to_invert(stack_paths, resolution, num_workers, port=port)

    stacks = get_stacks(stack_paths, invert_instructions)

    # Look for overlapping stacks
    combined_stacks = {k:v for k,v in stacks.items() if isinstance(v, list)}
    stacks = [v for k,v in stacks.items() if not isinstance(v, list)]

    logging.info(f'Found {len(combined_stacks)} combined stack')
    processed_combined_stacks = []
    if len(combined_stacks) > 0:
        logging.info('Checking groups of overlapping stacks')
        # Overlapping stacks were found, figure out overlapping regions
        for name_combined_stack, combined_stack in combined_stacks.items():
            # Detect overlapping regions
            remapped_tile_map, remapped_tile_invert = estimate_tile_map_positions(combined_stack, apply_gaussian, apply_clahe, scale=0.3)

            combined_stack = Stack()
            combined_stack.stack_name = name_combined_stack
            combined_stack._set_tilemaps_paths(remapped_tile_map)
            combined_stack.tile_maps_invert = remapped_tile_invert

            processed_combined_stacks.append(combined_stack)

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
    parser.add_argument('--dir_pattern',
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