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

from emalign.align_stack_z import align_stack_z
from emalign.utils.align_z_utils import compute_datasets_offsets
from emalign.utils.io_utils import get_ordered_datasets

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_dataset_z(config_path,
                    config_z_path,
                    num_workers, 
                    start_over):
    
    with open(config_path, 'r') as f:
        main_config = json.load(f)

    with open(config_z_path, 'r') as f:
        config_z = json.load(f)

    project_name    = main_config['project_name']
    output_path     = main_config['output_path']

    range_limit     = config_z['range_limit']    
    filter_size     = config_z['filter_size']    
    stride          = config_z['stride']
    patch_size      = config_z['patch_size']      
    scale_offset    = config_z['scale_offset']        
    scale_flow      = config_z['scale_flow']    
    step_slices     = config_z['step_slices']    

    datasets, z_offsets = get_ordered_datasets(output_path, project_name)
    offsets = compute_datasets_offsets(datasets, 
                                       z_offsets,
                                       range_limit,
                                       scale_offset, 
                                       filter_size,
                                       step_slices,
                                       num_workers)
    
    # Prepare the destination
    output_path = os.path.join(output_path, project_name)
    if not os.path.exists(output_path) or start_over:
        # Create container at destination if it doesn't exist or if user wants to start over
        # Shape destination starts as largest yx and last offset + shape of last dataset
        # yx could change shape based on warping but z should stay like this for this project
        shapes = np.array([dataset.shape for dataset in datasets])
        dest_shape = np.append(shapes[-1, 0] + offsets[-1, 0], shapes[:, 1:].max(0))
        destination = ts.open({'driver': 'zarr',
                               'kvstore': {
                                   'driver': 'file',
                                   'path': output_path,
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
        destination = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': output_path,
                                        }
                            },
                            dtype=ts.uint8
                            ).result()

    # For the first dataset, there is no first reference slice
    # script_path = os.path.abspath('/mnt/hdd1/SRC/alignment_sofima/new_scripts/align_stack_z.py')
    first_slice = None                
    for offset, dataset in tqdm(zip(offsets, datasets), 
                                total=len(datasets),
                                desc='Processing stacks'):
        
        config = {'destination_path': destination.kvstore.path,
                  'dataset_path': dataset.kvstore.path, 
                  'offset': offset.tolist(), 
                  'scale': scale_flow, 
                  'patch_size': patch_size, 
                  'stride': stride, 
                  'filter_size': filter_size,
                  'range_limit': range_limit,
                  'first_slice': first_slice,
                  'num_threads': num_workers}
        
        dataset_name = dataset.kvstore.path.split('/')[-2]
        output_dir = destination.kvstore.path.rsplit('/', maxsplit=3)[0]
        
        config_file = os.path.abspath(os.path.join(output_dir, dataset_name + '.json'))
        with open(config_file, 'w') as f:
            json.dump(config, f, indent='')
        
        # command = f'python {script_path} {config_file}'
        # p = subprocess.Popen(command.split(' '), 
        #                      env=os.environ, 
        #                      stdout=subprocess.PIPE)
        # for line in iter(p.stdout.readline, b''):
        #     print(line)
        # p.stdout.close()
        # p.wait()
        # p.communicate()  
        # p.terminate()  

        # if p.poll() is None:
        #     logging.info('Process did not terminate properly.')
        # else:
        #     logging.info('Process terminated successfully.')
        try:
            align_stack_z(destination.kvstore.path,
                          dataset.kvstore.path, 
                          offset, 
                          scale_flow, 
                          patch_size, 
                          stride, 
                          filter_size,
                          range_limit,
                          first_slice,
                          num_workers,
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
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        required=True,
                        type=str,
                        help='Path to the main task config.')
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    # align_dataset_z(config_path="/mnt/hdd1/SRC/alignment_sofima/output_alignment/praying_mantis/main_config.json",
    #                 config_z_path="/mnt/hdd1/SRC/alignment_sofima/new_scripts/config_z_align.json",
    #                 num_workers=25,
    #                 start_over=True)
