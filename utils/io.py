import json
import logging
import numpy as np
import os
import re
from scipy.fftpack import dst
import tensorstore as ts

from concurrent import futures
from cv2 import resize, GaussianBlur, createCLAHE
from glob import glob
from PIL import Image
from tqdm import tqdm

from sofima import warp


### FIND FILES

def get_tileset_resolution(tileset_path):
    '''
    Find resolution of a stack by reading an .info file located in the stack directory.
    '''
    info=None
    with os.scandir(tileset_path) as entries:
        for entry in entries:
            if entry.name.endswith('.info'):
                info = entry.path
                break
    
    if info is None:
        return None
    
    with open(info, 'r') as f:
        content = f.readlines()

    resolution = tuple(map(int, re.findall(r'\d+', content[5])))

    return (tileset_path, resolution)


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    
    # Get all directories containing tilesets that are present in main_dir    
    tileset_dirs = glob(main_dir + '/*/')

    stack_list = []
    # Find the ones with the right resolution
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = []
        for d in tileset_dirs:
            fs.append(tpe.submit(get_tileset_resolution, d))

        for f in tqdm(futures.as_completed(fs), total=len(fs), desc=f'Looking for resolution: {resolution}', leave=False):
            result = f.result()
            if result is None:
                continue
            # Find the directory with the right pattern if relevant
            for d in dir_pattern:
                if d in result[0].split('/')[-2] and result[1] == tuple(resolution):
                    stack_list.append(result[0])
    return sorted(stack_list)


### READ TIFS

def load_tilemap(tile_map_paths, invert, apply_gaussian, apply_clahe, scale):
    '''
    Load a tile map based on provided paths. Apply image processing if specified.
    '''
    z, tile_map_paths= list(tile_map_paths.items())[0]

    if not isinstance(invert, dict):
        invert = dict(zip(list(tile_map_paths.keys()), [invert]*len(tile_map_paths)))
    
    tile_map = {}
    tile_map_ds = {}
    for yx_pos, tile_path in tile_map_paths.items():
        img, img_ds = load_tif(tile_path, invert[yx_pos], apply_gaussian, apply_clahe, scale)
        tile_map[yx_pos] = img
        tile_map_ds[yx_pos] = img_ds

    # From first to last in order of acquisition. Defines which is on top (first in list)
    tile_map = {k: tile_map[k] for k in sorted(list(tile_map.keys()))}
    tile_map_ds = {k: tile_map_ds[k] for k in sorted(list(tile_map_ds.keys()))}

    return z, tile_map, tile_map_ds
    

def load_tif(tif_path, invert, apply_gaussian, apply_clahe, scale):
    '''
    Load tif using PIL.Image.open
    '''
    
    img = Image.open(tif_path)

    # Invert image
    if invert:
        img = np.invert(img).astype(np.uint8)
    else:
        img = np.array(img).astype(np.uint8)

    # Denoising: apply gaussian filter with kernel 3x3 and sigma 1
    if apply_gaussian:
        img = GaussianBlur(img,
                             (3,3),
                             1)
    # Contrast enhancement: apply CLAHE
    if apply_clahe:
        clahe = createCLAHE(clipLimit= 2,
                            tileGridSize= (10, 10))
        img = clahe.apply(img)

    # Downsample
    if scale < 1:
        return img, resize(img, None, fx=scale, fy=scale)

    return img, None


### READ TENSORSTORE 

def set_dataset_attributes(dataset, attrs):
    with open(os.path.join(dataset.kvstore.path, '.zattrs'), 'w') as f:
        json.dump(attrs, f, indent='')
    return True 

def get_dataset_attributes(dataset):
    with open(os.path.join(dataset.kvstore.path, '.zattrs'), 'r') as f:
        attrs = json.load(f)
    return attrs


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


def get_data_samples(dataset, step_slices, yx_target_resolution):

    resolution = np.array(get_dataset_attributes(dataset)['resolution'])[1:]

    z_max = dataset.domain.exclusive_max[0]-1

    z_list = np.arange(0, z_max, step_slices)
    z_list = np.append(z_list, z_max) if z_max not in z_list else z_list    

    data = []
    for z in z_list:
        arr = dataset[z].read().result()
        while not arr.any():
            z += 1
            arr = dataset[z].read().result()
        
        if np.any(resolution < yx_target_resolution):
            fy, fx = resolution/yx_target_resolution
            arr = resize(arr, None, fx=fx, fy=fy)
        elif np.any(resolution > yx_target_resolution):
            raise RuntimeError(f'Dataset resolution ({resolution.tolist()}) must be lower \
                               than target resolution ({yx_target_resolution.tolist()})')
        data.append(arr)

    return np.array(data)


### WRITE 

def render_slice_xy(dest, z, tile_map, meshes, stride, return_render=False, parallelism=1):
    try:
        stitched, _ = warp.render_tiles(tile_map, meshes, parallelism=parallelism, stride=(stride, stride))
        
        if return_render:
            return stitched
        else:
            y,x = stitched.shape
            if np.any(dest.domain.exclusive_max[1:] < np.array([y, x])):
                dest = dest.resize(exclusive_max=[None, y, x], expand_only=True).result()
            dest[z:z+1, :y, :x].write(stitched).result()
            return True
    except Exception as e:
        logging.error(f'Error in render_slice: {e}')
        raise


def render_slice_z(destination, z, data, inv_map, data_bbox, flow_bbox, stride, return_render=False, parallelism=1):

    aligned = warp.warp_subvolume(data, data_bbox, inv_map, flow_bbox, stride, data_bbox, 'lanczos', parallelism=parallelism)
    aligned = aligned[0,0,...]
    y,x = aligned.shape

    if return_render:
        return aligned
    else:
        if np.any(destination.domain.exclusive_max < np.array([z+1, y, x])):
            new_max = np.max([destination.domain.exclusive_max, [z+1, y, x]], axis=0)
            destination = destination.resize(exclusive_max=new_max, expand_only=True).result()

        destination[z:z+1, :y, :x].write(aligned).result()
        return True