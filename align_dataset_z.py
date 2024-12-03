import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'

# Influences performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import json
import logging
import numpy as np
import tensorstore as ts
import sys

from tqdm import tqdm

from .stack_align.align_stack_z import align_stack_z
from .utils.align_z import compute_datasets_offsets
from .utils.io import get_ordered_datasets

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_dataset_z(config_paths,
                    config_z_path,
                    num_workers, 
                    save_downsampled,
                    no_align,
                    start_over):

    # Combine the content of config files if multiple ones were provided and the projects match
    project_name = None
    output_path = None
    stack_configs = {}
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            main_config = json.load(f)

        if project_name is not None:
            assert project_name == main_config['project_name'], 'Project names between config files are not matching'
        else:
            project_name    = main_config['project_name']
        if output_path is not None:
            assert output_path == main_config['output_path'], 'Output paths between config files are not matching'
        else:
            output_path = main_config['output_path']

        stack_configs = stack_configs | main_config['stack_configs']

    with open(config_z_path, 'r') as f:
        config_z = json.load(f)

    range_limit     = config_z['range_limit']    
    filter_size     = config_z['filter_size']    
    patch_size      = config_z['patch_size']     
    stride          = config_z['stride']
    max_deviation   = config_z['max_deviation'] 
    max_magnitude   = config_z['max_magnitude']
    scale_offset    = config_z['scale_offset']        
    scale_flow      = config_z['scale_flow']    
    step_slices     = config_z['step_slices'] 

    yx_target_resolution = np.array(config_z['yx_target_resolution'])

    dataset_paths = [os.path.join(output_path, stack) for stack in stack_configs.keys()]
    datasets, z_offsets = get_ordered_datasets(dataset_paths)
    offsets = compute_datasets_offsets(datasets, 
                                       z_offsets,
                                       range_limit,
                                       scale_offset, 
                                       filter_size,
                                       step_slices,
                                       yx_target_resolution,
                                       num_workers)
    
    # Destination for the config files
    output_configs_path = os.path.join(os.path.dirname(output_path), 'config')
    os.makedirs(output_configs_path, exist_ok=True)
    
    # Prepare the destination
    project_output_path = os.path.join(output_path, project_name)
    if not os.path.exists(project_output_path) or start_over:
        logging.info(f'Creating project dataset at: \n    {project_output_path}\n')
        # Create container at destination if it doesn't exist or if user wants to start over
        # Shape destination starts as largest yx and last offset + shape of last dataset
        # yx could change shape based on warping but z should stay like this for this project
        shapes = np.array([dataset.shape for dataset in datasets])
        dest_shape = np.append(shapes[-1, 0] + offsets[-1, 0], shapes[:, 1:].max(0))
        destination = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': project_output_path,
                                           },
                               'metadata':{
                                   'shape': dest_shape,
                                   'chunks':[1,512,512]
                                           },
                               'transform': {'input_labels': ['z', 'y', 'x']}
                               },
                               dtype=ts.uint8, 
                               create=True,
                               delete_existing=True
                               ).result()
    else:
        logging.info(f'Opening existing project dataset at: \n    {project_output_path}\n')
        destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': project_output_path,
                                        }
                            },
                            dtype=ts.uint8
                            ).result()
        
    # Prepare the downsampled destination, which will serve for inspection
    ds_project_output_path = os.path.join(output_path, f'{save_downsampled}x_' + project_name)
    if not os.path.exists(ds_project_output_path) or start_over:
        logging.info(f'Creating downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        shapes = np.array([dataset.shape for dataset in datasets])//save_downsampled
        dest_shape = np.append(shapes[-1, 0] + offsets[-1, 0], shapes[:, 1:].max(0))
        ds_destination = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': ds_project_output_path,
                                           },
                               'metadata':{
                                   'shape': dest_shape,
                                   'chunks':[1,512,512]
                                           },
                               'transform': {'input_labels': ['z', 'y', 'x']}
                               },
                               dtype=ts.uint8, 
                               create=True,
                               delete_existing=True
                               ).result()
    else:
        logging.info(f'Opening existing downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        ds_destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': ds_project_output_path,
                                        }
                            },
                            dtype=ts.uint8
                            ).result()

    logging.info(f'Configuration files will be stored at: \n    {output_configs_path}\n')
    
    if no_align:
        pbar_desc = 'Preparing configuration files'
    else:
        pbar_desc = 'Processing stacks'

    # For the first dataset, there is no first reference slice
    first_slice = None                
    for offset, dataset in tqdm(zip(offsets, datasets), 
                                total=len(datasets),
                                desc=pbar_desc):
        
        config = {'destination_path': destination.kvstore.path,
                  'dataset_path': dataset.kvstore.path, 
                  'offset': offset.tolist(), 
                  'scale': scale_flow, 
                  'stride': stride, 
                  'patch_size': patch_size, 
                  'max_deviation': max_deviation,
                  'max_magnitude': max_magnitude,
                  'filter_size': filter_size,
                  'range_limit': range_limit,
                  'first_slice': first_slice,
                  'yx_target_resolution': yx_target_resolution.tolist(),
                  'save_downsampled': save_downsampled,
                  'num_threads': num_workers}
        
        dataset_name = dataset.kvstore.path.split('/')[-2]
        
        config_file = os.path.abspath(os.path.join(output_configs_path, 'z_' + dataset_name + '.json'))
        with open(config_file, 'w') as f:
            json.dump(config, f, indent='')

        if not no_align:
            try:
                align_stack_z(destination.kvstore.path,
                            dataset.kvstore.path, 
                            offset, 
                            scale_flow, 
                            patch_size, 
                            stride, 
                            max_deviation,
                            max_magnitude,
                            filter_size,
                            range_limit,
                            first_slice,
                            yx_target_resolution,
                            num_workers,
                            save_downsampled,
                            start_over)
            except Exception as e:
                raise RuntimeError(e)
        
        # Set the last slice of this dataset to be the reference for the next dataset
        first_slice = int(offset[0] + dataset.shape[0] - 1)
            
    logging.info('Done!')
    logging.info(f'Output: {output_path}')


if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in Z based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment).\n\
                                   The dataset must have been aligned in XY and written to a zarr container, before using this script.\n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATHS',
                        dest='config_paths',
                        required=True,
                        nargs='+',
                        type=str,
                        help='Path to the main task configs. \
                              Can provide one or multiple configs with the same project name to align all stacks they point to')
    parser.add_argument('-cfg-z', '--config-z',
                        metavar='CONFIG_Z_PATH',
                        dest='config_z_path',
                        required=True,
                        type=str,
                        help='Path to the z alignment task config.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=1,
                        help='Number of threads to use for processing. Default: 0 (all cores available)')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=4,
                        help='Factor to use for downsampling the dataset that will be saved for inspection. Default: 4')
    parser.add_argument('--no-align',
                        dest='no_align',
                        default=False,
                        action='store_true',
                        help='Do not align and only create configuration files. Default: False')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Deletes existing output dataset and start over. Default: False')

    args=parser.parse_args()

    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}')

    align_dataset_z(**vars(args))   