import argparse
import json
import neuroglancer
import numpy as np
import os
import tensorstore as ts

from glob import glob
from time import sleep

from emalign.utils.io import get_ordered_datasets


def read_data(
            dataset_path,
            data_range=None,
            keep_missing=False):
    
    spec = {'driver': 'zarr',
                        'kvstore': {
                                'driver': 'file',
                                'path': dataset_path,
                                    }}
    dataset = ts.open(spec,
                      read=True,
                      dtype=ts.uint8
                      ).result()
    
    if data_range is None:
        data = dataset[:].read().result()
    else:
        z0 = data_range[0]
        if len(data_range) == 2:
            # Bound range to the possible values
            z1 = min(data_range[1], dataset.domain.exclusive_max[0])
        elif len(data_range) == 1:
            # Only one value, means we go from that value to the end
            z1 = dataset.domain.exclusive_max[0]
        data = dataset[z0:z1].read().result()

    if not keep_missing:
        data = data[data.any(axis=(1,2))]
    
    return data


def show_data(layers,
              port=55555):

    neuroglancer.set_server_bind_address(bind_address='localhost', bind_port=port)
    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=[50, 50, 50])
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.dimensions = dimensions

        for i, (name, data) in enumerate(layers.items()):
            layer = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(data.T, dimensions))
            s.layers[name] = layer
            s.layers[name].visible = i == 0
        s.layout = 'xy'

    url = viewer.get_viewer_url()
    print(url)
    input('Press ENTER to quit')


def inspect_dataset(
            dataset_path,
            data_range=[0],
            keep_missing=False,
            project_configs=[],
            mode=None,
            port=55555):
    
    modes = ['z_transitions', 'all_ds']
    
    if mode is not None and mode not in modes:
        raise ValueError(f'Invalid mode. Must be one of: {modes}')
    
    dataset_name = os.path.basename(os.path.abspath(dataset_path))
    
    if mode is None:
        data = {dataset_name: read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)}    
    elif mode == 'z_transitions':
        dataset_paths = []
        config_paths = glob(os.path.join(project_configs, '*.json'))

        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            dataset_paths.append(os.path.join(config['dataset_path']))

        _, z_offsets = get_ordered_datasets(dataset_paths)

        window = 20
        data = {}
        for z, _, _ in z_offsets:
            data_range = [int(z - max(1, window/2)), int(z + max(1, window/2))]

            try:
                data.update({f'{dataset_name}_{z}': read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)})
            except:
                continue
    elif mode == 'all_ds':
        dataset_paths = sorted(glob(os.path.join(dataset_path, '*')))

        data = {}
        for dataset_path in dataset_paths:
            dataset_name = os.path.basename(dataset_path)
            data.update({dataset_name: read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)})     

    show_data(data, port=port)


if __name__ == '__main__':

    parser=argparse.ArgumentParser('Inspect image data stored in a zarr container.')
    
    parser.add_argument('-d', '--dataset-path',
                        metavar='DATASET_PATH',
                        dest='dataset_path',
                        required=True,
                        type=str,
                        default=None,
                        help='Path to the zarr container containing the final alignment.')
    parser.add_argument('--data-range',
                        metavar='DATA_RANGE',
                        dest='data_range',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='Range of slice indices to show. One value will be consider as the lower bound.' \
                             'If too high, will be bounded to the max possible value.')
    parser.add_argument('--keep-missing',
                        dest='keep_missing',
                        default=False,
                        action='store_true',
                        help='Keep missing slices as black images. Default: False')
    parser.add_argument('-cfg', '--config',
                        metavar='PROJECT_CONFIGS',
                        dest='project_configs',
                        required=False,
                        # nargs='+',
                        type=str,
                        help='Path to the project configs containing information about the dataset\'s transitions.')
    parser.add_argument('--mode',
                        dest='mode',
                        type=str,
                        default=None,
                        help='Visualization mode. One of: z_transitions, all_ds')
    parser.add_argument('--port',
                        metavar='PORT',
                        dest='port',
                        required=False,
                        type=int,
                        default=55555,
                        help='Port to use for neuroglancer.')
    
    
    args=parser.parse_args()

    inspect_dataset(**vars(args))
