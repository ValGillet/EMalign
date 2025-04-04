import argparse
import json
import neuroglancer
import numpy as np
import os
import tensorstore as ts

from glob import glob

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
        data_range[1] = min(data_range[1], dataset.domain.exclusive_max[0])
        data = dataset[data_range[0]:data_range[1]].read().result()

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

        for name, data in layers.items():
            s.layers[name] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(data.T, dimensions), cross_section_render_scale = 1)
        s.layout = 'xy'

    url = viewer.get_viewer_url()
    print(url)
    input('Press ENTER to quit')


def inspect_dataset(
            dataset_path,
            data_range=None,
            keep_missing=False,
            project_configs=[],
            transitions_only=False,
            print_shape=False,
            port=55555):
    
    dataset_name = os.path.basename(os.path.abspath(dataset_path))

    if print_shape:
        spec = {'driver': 'zarr',
                        'kvstore': {
                                'driver': 'file',
                                'path': dataset_path,
                                    }}
        dataset = ts.open(spec,
                        read=True,
                        dtype=ts.uint8
                        ).result()
        print(f'Dataset shape (ZYX): {dataset.shape}')
        return
    
    if transitions_only:
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
                data.update({f'{dataset_name}_{z}': read_data(dataset_path, data_range=data_range, keep_missing=keep_missing)})
            except:
                continue
    else:
        data = {dataset_name: read_data(dataset_path, data_range=data_range, keep_missing=keep_missing)}

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
                        nargs=2,
                        type=int,
                        help='Range of slice indices to show.')
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
    parser.add_argument('--transitions-only',
                        dest='transitions_only',
                        default=False,
                        action='store_true',
                        help='Only show the transitions between parts of the dataset, as opposed to the full thing.')
    parser.add_argument('--port',
                        metavar='PORT',
                        dest='port',
                        required=False,
                        type=int,
                        default=55555,
                        help='Port to use for neuroglancer.')
    parser.add_argument('--print-shape',
                        dest='print_shape',
                        default=False,
                        action='store_true',
                        help='Print the shape of this dataset. Useful to know what possible data range can be shown.')
    
    
    args=parser.parse_args()

    inspect_dataset(**vars(args))