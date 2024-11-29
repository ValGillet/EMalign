import argparse
import numpy as np
import os
import tensorstore as ts

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.transform import pyramid_gaussian
from tqdm import tqdm


'''
Convert a zarr container to image pyramids
'''


def write_pyramid_tiles(pyramid, z, output_path, tile_shape):

    tile_dir = os.path.join(output_path, str(z))
    os.makedirs(tile_dir, exist_ok=True)

    for ds_factor, data in enumerate(pyramid):
        level_dir = os.path.join(tile_dir, str(ds_factor))
        os.makedirs(level_dir, exist_ok=True)

        if data.dtype != np.uint8:
            data = 255 * data
            data = data.astype(np.uint8)

        (tiles_y, tiles_x), re = np.divmod(data.shape, tile_shape)
        diff = (tile_shape - re) * re.astype(bool)

        # Pad the data so tile_shape is a factor of its shape
        data = np.pad(data, np.stack([(0,0), diff.astype(int)]).T)

        # Create tiles
        tiles = {}
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile = data[y*tile_shape:(y+1)*tile_shape, x*tile_shape:(x+1)*tile_shape]
                tiles.update({f'{x}_{y}.jpg': tile})
    
        # Write tiles
        for filename, tile in tiles.items():
            try:
                im = Image.fromarray(tile)
                im.save(os.path.join(level_dir, filename), 'JPEG') 
            except Exception as e:
                raise RuntimeError(e)
    
    return True


def dataset_to_pyramid(dataset_path, 
                       output_path, 
                       max_layer=3, 
                       tile_shape=1024, 
                       downsample_factor=2, 
                       num_threads=0,
                       duplicate_missing_slices=True):

    dataset = ts.open({'driver': 'zarr',
                            'kvstore': {
                                    'driver': 'file',
                                    'path': dataset_path,
                                        }
                      }).result()

    fs_read = []
    fs_write = []
    with ThreadPoolExecutor(num_threads) as tpe:
        for z in range(dataset.shape[0]):
            fs_read.append(tpe.submit(lambda z=z: dataset[z].read().result()))

        fs_read = fs_read[::-1]

        for z in tqdm(range(dataset.shape[0]), desc='Computing pyramid levels', leave=True):
            data = fs_read.pop().result()

            if data.any():
                pyramid = tuple(pyramid_gaussian(data, downscale=downsample_factor, max_layer=max_layer))
            elif not duplicate_missing_slices:
                raise RuntimeError('Missing slice.')

            fs_write.append(tpe.submit(write_pyramid_tiles, pyramid, z, output_path, tile_shape))

        for f in tqdm(as_completed(fs_write), desc='Finishing up writing tasks', leave=True):
            try:
                f.result()
            except Exception as e:
                raise RuntimeError(e)


if __name__ == '__main__':

    parser=argparse.ArgumentParser('Script aligning tiles in Z based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment).\n\
                                   The dataset must have been aligned in XY and written to a zarr container, before using this script.\n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')
    parser.add_argument('-i', '--input',
                        metavar='DATASET_PATH',
                        dest='dataset_path',
                        required=True,
                        type=str,
                        help='Path to the aligned dataset to convert to pyramid.')
    parser.add_argument('-o', '--output',
                        metavar='OUTPUT_PATH',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='Path to the directory where to write the pyramid levels.')
    parser.add_argument('-l', '--max-layer',
                        metavar='MAX_LAYER',
                        dest='max_layer',
                        type=int,
                        default=3,
                        help='Maximum layer of the pyramid, corresponding to how many times the data was downsampled. Default: 3')
    parser.add_argument('--tile-shape',
                        metavar='TILE_SHAPE',
                        dest='tile_shape',
                        type=int,
                        default=1024,
                        help='Shape of the tiles making up the pyramid, in pixels. Default: 1024')
    parser.add_argument('-ds', '--downsample-factor',
                        metavar='SCALE',
                        dest='downsample_factor',
                        type=int,
                        default=2,
                        help='Factor to use for downsampling the dataset between each level of the pyramid. Default: 2')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_threads',
                        type=int,
                        default=0,
                        help='Number of threads to use for processing. Default: 0 (all cores available)')
    parser.add_argument('--error-missing',
                        dest='duplicate_missing_slices',
                        default=True,
                        action='store_false',
                        help='Raises an error if a missing slice is encountered. Default: The previous slice is duplicated')



    args=parser.parse_args()

    dataset_to_pyramid(**vars(args))