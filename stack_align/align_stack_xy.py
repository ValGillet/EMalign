import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     /home/autoseg/anaconda3/envs/alignment/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. 
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called")

import json
import logging
import numpy as np
import tensorstore as ts

from concurrent import futures
from tqdm import tqdm

from emalign.utils.stacks import Stack, parse_stack_info
from emalign.utils.io import *
from emalign.utils.align_xy import *
from emalign.utils.inspect import *
from emalign.utils.offsets import estimate_tilemap_overlap

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def align_stack_xy(output_path,
                   stack_name,
                   tile_maps_paths,
                   tile_maps_invert,
                   resolution,
                   offset,
                   stride,
                   prelim_overlap,
                   apply_gaussian,
                   apply_clahe,
                   num_cores):
    
    '''
    Align and stitch image stack in XY. 

    Args:

        output_path (``str``):

            Path to the zarr container where the stack will be written.

        stack_name (``str``):

            Name of the stack. Will be used as the dataset name in the destination zarr container. 

        tile_maps_paths (``dict``):

            Dictionnary of slices to dictionnary of tile grid positions to paths of tifs.

        tile_maps_invert (``dict``):

            Dictionnary of tile grid positions to boolean, describing whether tiles need to be inverted at that position. 

        resolution (list of ``int``):

            List of 2 int corresponding to the YX resolution in nanometers.

        offset (list of ``int``):

            List of 3 int corresponding to the ZYX offset of the stack in voxels.

        stride (``int``):

            YX stride for computing the elastic mesh, in pixels. 

        prelim_overlap (``int``):

            Likely overlap between tiles. Overlap will be finely determined within a window given by this value.

        apply_gaussian (``bool``):

            Whether to apply a gaussian filter to tiles for denoising.

        apply_clahe (``bool``):

            Whether to apply CLAHE to tiles to enhance contrast.

        num_cores (``int``):

            Number of CPUs to use for rendering stitched images.
    '''
    
    stack = Stack(stack_name=stack_name, 
                  tile_maps_paths=tile_maps_paths, 
                  tile_maps_invert=tile_maps_invert)

    # Variables
    zarr_path  = os.path.join(output_path, stack.stack_name)
    attrs_path = os.path.join(zarr_path, '.zattrs')

    z_offset = min(stack.slices)
    z_shape  = max(stack.slices)-min(stack.slices)
    offset[0] = z_offset

    # Skip if already fully processed
    if os.path.exists(attrs_path):
       logging.info(f'Skipping {stack.stack_name} because it was already processed.')
       return False

    dataset = ts.open({'driver': 'zarr',
                        'kvstore': {
                            'driver': 'file',
                            'path': zarr_path,
                                    },
                        'metadata':{
                            'shape': [z_shape + 1, 
                                        1, 1],
                            'chunks':[1,512,512]
                                    },
                        'transform': {'input_labels': ['z', 'y', 'x']}
                        },
                        dtype=ts.uint8, 
                        create=True,
                        delete_existing=True).result()   
    
    #####################
    ### PROCESS STACK ###
    #####################
    overlap_pad = 50
    pbar = tqdm(stack.slices, position=2, desc=f'{stack.stack_name}: Processing', leave=False)
    for z in pbar:
        pbar.set_description(f'{stack.stack_name}: Loading tile_map...')
        z, tile_map, _ = load_tilemap({z: stack.slice_to_tilemap[z]}, 
                                        stack.tile_maps_invert,
                                        apply_gaussian, 
                                        apply_clahe,
                                        1)
        tile_space = (np.array(list(tile_map.keys()))[:,1].max()+1, 
                      np.array(list(tile_map.keys()))[:,0].max()+1)
        
        # Predict the overlap between tiles
        pbar.set_description(f'{stack.stack_name}: Estimating overlap...')
        overlap = estimate_tilemap_overlap(tile_space,
                                           tile_map,
                                           preliminary_overlap=prelim_overlap,
                                           scale=[0.3,0.5])
        max_overlap = max([t.shape for t in tile_map.values()])
        while overlap == 0:
            # The preliminary overlap is likely too small
            # Increase it until we find a value, or max overlap is too high
            prelim_overlap += 100

            if prelim_overlap >= max_overlap:
                raise RuntimeError('Maximum overlap reached. Images may not overlap, or scale at which to search for overlap is too small.')
            overlap = estimate_tilemap_overlap(tile_space,
                                           tile_map,
                                           preliminary_overlap=prelim_overlap,
                                           scale=[0.3,0.5])

        if len(tile_map) > 1:
            # There are more than one tiles
            # Pad tiles so they are all the same shape (required by sofima)
            max_shape = np.max([t.shape for t in tile_map.values()],axis=0)

            tile_masks = {}
            for k in sorted(tile_map): 
                tile = tile_map[k]
                mask = np.ones_like(tile)

                if np.any(np.array(tile.shape) != max_shape):
                    d = k[::-1] == (np.array(tile_space) - 1)
                    d[1] = np.logical_not(d[1])

                    tile = pad_to_shape(tile, max_shape, d.astype(int))
                    mask = pad_to_shape(mask, max_shape, d.astype(int))
                
                tile_map[k] = tile
                tile_masks[k] = mask
            
            pbar.set_description(f'{stack.stack_name}: Computing elastic meshes...')
            cx, cy, coarse_mesh = get_coarse_offset(tile_map, 
                                                    tile_space,
                                                    overlap=[overlap,               # try first
                                                             overlap+overlap_pad]   # try second
                                                   )

            patch_size = 160
            if overlap > patch_size:
                meshes = get_elastic_mesh(tile_map, 
                                          cx, 
                                          cy, 
                                          coarse_mesh,
                                          stride=stride,
                                          patch_size=patch_size)
                render_stride=stride
            else:
                meshes = get_elastic_mesh(tile_map, 
                                          cx, 
                                          cy, 
                                          coarse_mesh,
                                          stride=10,
                                          patch_size=40)
                render_stride=10

            # Determine margin by finding the minimum displacement in X or Y between adjacent tiles
            # Margin is how many pixels to ignore from the tiles when rendering. Too high leaves a delimitation, too low leaves a gap
            min_displacement = np.abs(np.concatenate([cx[0,0,0,:][~np.isnan(cx[0,0,0,:])], 
                                                      cy[1,0,0,:][~np.isnan(cy[1,0,0,:])]])).min()
            margin = max(int(min_displacement // 2 - 5), 1)
            
            # Ensure that first tiles acquired are rendered last because they are sharper and should be on top
            meshes = {k:meshes[k] for k in sorted(meshes)[::-1]}
            pbar.set_description(f'{stack.stack_name}: Rendering...')
            render_slice_xy(dataset, z-z_offset, tile_map, meshes, render_stride, tile_masks, parallelism=num_cores, margin=margin)
        else:
            # There is only one tile, no need to compute anything
            pbar.set_description(f'{stack.stack_name}: Writing unique tile...')
            render_slice_xy(dataset, z-z_offset, tile_map, None, None, None, parallelism=num_cores)

    pbar.set_description(f'{stack.stack_name}: done')

    # Attributes are ZYX coordinates
    # Resolution in Z is hard coded to be 50 nm currently
    # Keys are used in subsequent steps in the alignment and segmentation pipeline
    attributes = {'voxel_offset': offset,
                  'offset': list(map(int, np.array(offset)*np.array([50, *resolution]))),
                  'resolution': list(map(int, (50, *resolution)))}

    set_dataset_attributes(dataset, attributes)

    return True


if __name__ == '__main__':

    config_path = sys.argv[1]
    stack_name  = sys.argv[2]
    num_cores = int(sys.argv[3])

    with open(config_path, 'r') as f:
        main_config = json.load(f)

    main_dir        = main_config['main_dir']
    output_path     = main_config['output_path']
    resolution      = main_config['resolution']
    offset          = main_config['offset']
    stride          = main_config['stride']
    prelim_overlap  = main_config['overlap']
    apply_gaussian  = main_config['apply_gaussian']
    apply_clahe     = main_config['apply_clahe']
    stack_configs   = main_config['stack_configs']
    
    tile_maps_paths, tile_maps_invert = parse_stack_info(stack_configs[stack_name])

    align_stack_xy(output_path=output_path,
                   stack_name=stack_name,
                   tile_maps_paths=tile_maps_paths,
                   tile_maps_invert=tile_maps_invert,
                   resolution=resolution,
                   offset=offset,
                   stride=stride,
                   prelim_overlap=prelim_overlap,
                   apply_gaussian=apply_gaussian,
                   apply_clahe=apply_clahe,
                   num_cores=num_cores)
