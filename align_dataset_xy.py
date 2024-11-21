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
import tensorstore as ts

from concurrent import futures
from tqdm import tqdm

from new_utils.stacks_utils import Stack
from new_utils.io_utils import *
from new_utils.align_xy_utils import *
from new_utils.check_utils import *
from align_stack_xy import align_stack_xy


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def parse_stack_info(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    tile_maps_paths = {}

    for z, tm in config['tile_maps'].items():
        tm = {tuple(int(i) 
                for i in re.findall(r'\b\d+\b', k)): v for k,v in tm.items()}
        tile_maps_paths.update({int(z): tm})

    tile_maps_invert = {tuple(int(i) for i in re.findall(r'\b\d+\b', k)): v 
                            for k,v in config['tile_maps_invert'].items()}
    return tile_maps_paths, tile_maps_invert


def align_dataset_xy(config_path,
                     num_workers):
    
    with open(config_path, 'r') as f:
        main_config = json.load(f)

    main_dir        = main_config['main_dir']
    output_path     = main_config['output_path']
    resolution      = main_config['resolution']
    scale           = main_config['scale']
    stride          = main_config['stride']
    overlap         = main_config['overlap']
    apply_gaussian  = main_config['apply_gaussian']
    apply_clahe     = main_config['apply_clahe']
    stack_configs   = main_config['stack_configs']

    if not output_path.endswith('.zarr'):
        raise RuntimeError('Output path must be a zarr container (.zarr)')

    # Find tilesets with wanted resolution
    logging.info(f'Aligning tilesets found in: {main_dir}')
    logging.info(f'Destination: {output_path}')
    logging.info(f' - Resolution: {resolution}')
    logging.info(f' - Compute scale: {scale}')
    logging.info(f' - Tile overlap (px): {overlap}')
    logging.info(f' - Apply gaussian: {apply_gaussian}')
    logging.info(f' - Apply CLAHE: {apply_clahe}')
    logging.info(f'Will align {len(stack_configs)} tilesets, including {main_config['tilesets_combined']} combined.')
    for s in stack_configs.keys():
        logging.info(f'    {s}')

    for stack_name, stack_config_path in tqdm(stack_configs.items(), 
                                                total=len(stack_configs), 
                                                position=1, 
                                                desc='Processing stacks', 
                                                leave=True):
        tile_maps_paths, tile_maps_invert = parse_stack_info(stack_config_path)
        align_stack_xy(output_path,
                       stack_name,
                       tile_maps_paths,
                       tile_maps_invert,
                       resolution,
                       stride,
                       overlap,
                       scale,
                       apply_gaussian,
                       apply_clahe,
                       num_workers)
    logging.info('Done!')
    logging.info(f'Output: {output_path}')        
    

if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        required=True,
                        type=str,
                        help='Path to the main task config.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=1,
                        help='Number of threads to use for processing. Default: 0 (all cores available)')
    args=parser.parse_args()


    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}')

    align_dataset_xy(**vars(args))    