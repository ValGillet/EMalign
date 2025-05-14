import argparse
import json
import logging
import numpy as np
import os
import sys

from glob import glob
from time import sleep
from tqdm import tqdm

from emalign.align_z.align_z import align_arrays_z
from emalign.align_z.utils import compute_datasets_offsets, get_ordered_datasets


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def prep_config_z(config_paths,
                  config_z_path,
                  num_workers, 
                  port):
    
    with open(config_z_path, 'r') as f:
        config_z = json.load(f)

    dataset_paths = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            main_config = json.load(f)

        project_name    = main_config['project_name']
        output_path     = main_config['output_path']

        stride          = config_z['stride']
        patch_size      = config_z['patch_size']    
        max_deviation   = config_z['max_deviation']
        max_magnitude   = config_z['max_magnitude']  
        scale_offset    = config_z['scale_offset']        
        scale_flow      = config_z['scale_flow']    
        step_slices     = config_z['step_slices']
        filter_size     = config_z['mask_filter_size']    
        range_limit     = config_z['mask_range_limit']
        yx_target_resolution = config_z['yx_target_resolution']
        k0      = config_z['k0'] 
        k       = config_z['k'] 
        gamma   = config_z['gamma']

        destination_path = os.path.join(output_path, project_name)
        config_dir = os.path.join(os.path.dirname(output_path), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        dataset_paths += [d for d in glob(os.path.join(output_path, '*/')) if os.path.basename(d[:-1]) != project_name]
    datasets, z_offsets = get_ordered_datasets(dataset_paths)

    existing_configs = glob(os.path.join(config_dir, 'z_*.json'))
    if len(existing_configs) > 0:
        logging.info('Config already exists in dir, exiting process...')
        sys.exit()
    
    pad_offset = (500,500)
    offsets = compute_datasets_offsets(datasets, 
                                       z_offsets,
                                       range_limit,
                                       scale_offset, 
                                       filter_size,
                                       step_slices,
                                       yx_target_resolution,
                                       pad_offset,
                                       num_workers)

    reference = None
    first_slice_z = None
    aligned_slices = []
    configs = []
    for dataset, offset in tqdm(zip(datasets, offsets), desc='Aligning regions of transition'):
        if reference is not None:
            i = 0
            curr = dataset[i].read().result()
            while not curr.any():
                i += 1
                curr = dataset[i].read().result()
            curr = np.pad(curr, np.stack([offset[1:], (0,0)]).T)
            
            unaligned, aligned = align_arrays_z(reference, curr, scale_flow,
                                                patch_size, stride, max_magnitude, max_deviation,
                                                filter_size, range_limit, 
                                                k0, k, gamma,
                                                num_workers)

            aligned_slices.append([unaligned, aligned])

        first_slice_z = None if reference is None else int(offset[0] + dataset.shape[0] - 1)
        config = {'destination_path': destination_path,
                  'dataset_path': dataset.kvstore.path, 
                  'offset': offset.tolist(), 
                  'scale': scale_flow, 
                  'patch_size': patch_size, 
                  'stride': stride, 
                  'max_deviation': max_deviation,
                  'max_magnitude': max_magnitude,
                  'k0': k0,
                  'k': k,
                  'gamma': gamma,
                  'filter_size': filter_size,
                  'range_limit': range_limit,
                  'first_slice': first_slice_z,
                  'num_threads': num_workers}
        
        dataset_name = dataset.kvstore.path.split('/')[-2]
        
        config_path = os.path.abspath(os.path.join(config_dir, 'z_' + dataset_name + '.json'))
        if reference is None:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent='')
        else:
            configs.append((config_path, config))
        
        # Get the reference slice of the next dataset in line
        reference = dataset[dataset.domain.exclusive_max[0] - 1].read().result()

        i = 1
        while not reference.any():
            i += 1
            reference = dataset[dataset.domain.exclusive_max[0] - i].read().result()

        reference = np.pad(reference, np.stack([offset[1:], (0,0)]).T)

    logging.info('Test alignment done, please check the result')
    logging.info(' - z=0: Reference slice.')
    logging.info(' - z=1: Moving slice.')

    pbar = tqdm(enumerate(aligned_slices), position=0)
    for i, data in pbar:
        ref_dataset = datasets[i].kvstore.path.split('/')[-2]
        moving_dataset = datasets[i+1].kvstore.path.split('/')[-2]
        transition = f'{moving_dataset} to {ref_dataset}'
        pbar.set_description(transition)

        config_path, config = configs[i]
        stride = config['stride']
        patch_size = config['patch_size']
        unaligned, aligned = data
        while True:
            answer = display_array(aligned, unaligned, transition, port)
            if answer == 'y':
                break
            else:
                print('\nProvide new values for alignment parameters (increase to increase deformation of the moving image)')
                print('Stride should be patch_size//4 or patch_size//2')
                patch_size = int(input(f'Patch size (current value = {patch_size}): '))
                stride = int(input(f'Stride (current value = {stride}): '))

                logging.info('Computing alignment with the new values...')
                reference, curr = unaligned
                unaligned, aligned = align_arrays_z(reference, curr, scale_flow,
                                                    patch_size, stride, max_magnitude, max_deviation,
                                                    filter_size, range_limit, 
                                                    k0, k, gamma,
                                                    num_workers)

        config['stride'] = stride
        config['patch_size'] = patch_size

        with open(config_path, 'w') as f:
            json.dump(config, f, indent='')

def display_array(aligned, unaligned, name, port):
    import neuroglancer
    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=port)
    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=[50, 50, 50])
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.dimensions = dimensions
        s.position = np.append([0], np.array(aligned.shape[:1])//2)
        s.layers[name + '_unaligned'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(unaligned.T, dimensions), cross_section_render_scale = 1)
        s.layers[name + '_aligned'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(aligned.T, dimensions), cross_section_render_scale = 1)
        s.layout = 'xy'

    url = viewer.get_viewer_url()
    print('\nhttp://localhost:' + url.split(f':')[-1])
    
    a = input('Is the Z alignment satisfying? (y/n) ')
    sleep(1)
    return a
                
if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-cfg', '--config_path',
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        required=True,
                        type=str,
                        help='Path to the project config file.')
    parser.add_argument('-cfg-z', '--config-z',
                        metavar='CONFIG_Z',
                        dest='config_z_path',
                        required=True,
                        type=str,
                        help='Path to the config file containing the parameters relevant for z alignment.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=None,
                        help='Number of cores to use for multiprocessing and multithreading. Default: 0 (all cores available)')
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

    prep_config_z(**vars(args))    