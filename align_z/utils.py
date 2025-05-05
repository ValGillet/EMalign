''' Utilities for alignment of stacks along Z axis.'''

import numpy as np
import tensorstore as ts

from concurrent import futures
from tqdm import tqdm

from emprocess.utils.io import get_dataset_attributes
from emprocess.utils.transform import rotate_image

from ..io.store import get_data_samples
from ..array.sift import estimate_transform_sift


def get_ordered_datasets(dataset_paths):

    dataset_stores = []
    offsets = []
    for ds in dataset_paths:
        spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': ds,
                }
               }
        dataset = ts.open(spec).result()
        dataset_stores.append(dataset)

        attrs = get_dataset_attributes(dataset)
        offsets.append(attrs['voxel_offset'])

    offsets = np.array(offsets)

    # Make sure that datasets come in the right order (offsets)
    dataset_stores = [dataset_stores[i] for i in np.argsort(offsets[:, 0])]
    offsets = offsets[np.argsort(offsets[:, 0])]
    return dataset_stores, offsets


def compute_datasets_offsets(datasets, 
                             offsets,
                             scale, 
                             step_slices,
                             yx_target_resolution,
                             pad_offset=(0,0),
                             num_workers=0):
    
    offsets_yx = [[0,0]]
    fs = []
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        for dataset in datasets:
            fs.append(tpe.submit(get_data_samples, dataset, step_slices, yx_target_resolution))

        fs = fs[::-1]

        # Do very first dataset
        data = fs.pop().result()
        inner_offsets = [estimate_transform_sift(data[i-1], data[i], scale=scale, refine_estimate=True)[0] 
                        for i in range(1, len(data))]
        # Offset between first and last image of the first dataset
        last_inner_offset = np.sum(inner_offsets, axis=0)
        rotation_angle = 0
        for _ in tqdm(range(len(fs)),
                      desc=f'Calculating offset between {len(datasets)} datasets.'):
            # Reference is the latest image before the current dataset
            prev = rotate_image(data[-1], rotation_angle)
            data = fs.pop().result()

            offset_to_last, rotation_angle, _ = estimate_transform_sift(prev, data[0], scale=scale, refine_estimate=True)
            # Calculate offset to the last stack 
            offsets_yx.append(offset_to_last + last_inner_offset)

            # Offset between first and last image (to account for differences between first images of different stacks and drift)
            last_inner_offset = np.sum([estimate_transform_sift(data[i-1], data[i], scale=scale)[0] 
                                    for i in range(1, len(data))], axis=0)

    offsets_yx = np.array(offsets_yx)

    yx_cumsum = np.cumsum(offsets_yx, axis=0)
    offsets[:, 1:] += (yx_cumsum - np.min(yx_cumsum, axis=0)).astype(int)
    offsets[:, 1:] += np.array(pad_offset)
    return offsets