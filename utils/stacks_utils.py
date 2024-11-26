
import numpy as np

from glob import glob
from collections import defaultdict

from .align_xy_utils import estimate_offset_horiz, estimate_offset_vert, test_laplacian
from .io_utils import *


class Stack:
    def __init__(self, stack_path=None, stack_name=None, tile_maps_paths=None, tile_maps_invert=None):
        self.stack_path = stack_path
        
        if stack_path is not None and stack_name is None:
            self.stack_name = stack_path.split('/')[-2]
        else:
            self.stack_name = stack_name

        if tile_maps_paths is not None:
            self._set_tilemaps_paths(tile_maps_paths)
        
        if tile_maps_invert is not None:
            self.tile_maps_invert = tile_maps_invert
            
    def __str__(self):
        return self.stack_name

    def _get_yx_pos(self, n):
        # Parse YX position relative to other tiles from name. Based on Volumescope's MAPs naming scheme.
        xy_pos = n.split('Tile_')[-1][:7]
        return tuple(int(i)-1 for i in xy_pos.split('-'))[::-1]

    def _get_slice(self, n):
        return int(n.split('s')[-1][:4])

    def _get_tilemaps_paths(self):
        # Produces lists of paths of all tifs contained in self.stack_path
        tile_paths = glob(self.stack_path + '*.tif')

        # Get paths and group by slice
        self.slice_to_paths = defaultdict(list)
        for tile_path in tile_paths:
            self.slice_to_paths[self._get_slice(tile_path)].append(tile_path)

        self.slices = sorted(list(self.slice_to_paths.keys()))
        
        # Sort
        self.slice_to_paths = {k: self.slice_to_paths[k] for k in self.slices}

        # Prep tilemaps
        self.slice_to_tilemap = defaultdict(dict)
        tile_indices = set()
        for s in self.slices:
            d = {}
            for t in self.slice_to_paths[s]:
                tile_indices.add(self._get_yx_pos(t))
                d.update({self._get_yx_pos(t): t for t in self.slice_to_paths[s]})
            self.slice_to_tilemap.update({s: d})   

        self.tile_maps_invert = dict(zip(tile_indices, [None]*len(tile_indices)))
        
    def _set_tilemaps_paths(self, tile_map_paths):

        self.slice_to_tilemap = tile_map_paths
        self.slices = sorted(list(self.slice_to_tilemap.keys()))
        self.slice_to_tilemap = {k: self.slice_to_tilemap[k] for k in self.slices}
        
        self.slice_to_paths = defaultdict(list)
        for z, d in self.slice_to_tilemap.items():
            self.slice_to_paths[z].append(list(d.values()))

        tile_indices = list(d.keys())
        self.tile_maps_invert = dict(zip(tile_indices, [None]*len(tile_indices)))


# COMBINE STACKS
def get_tile_maps_offset(tile_map_1, tile_map_2, overlap):
    # all xy coordinates
    coords_1 = np.array(list(tile_map_1.keys())) 
    coords_2 = np.array(list(tile_map_2.keys()))

    best_result = 0
    # 1 is top
    # last row
    for k1 in coords_1[coords_1[:, 1] == coords_1[:, 1].max()]:
        # first row
        for k2 in coords_2[coords_2[:, 1] == 0]:
            
            img1 = tile_map_1[tuple(k1)]
            img2 = tile_map_2[tuple(k2)]
    
            # img1 is top, img2 is bot
            xy_offset, top, bot = estimate_offset_vert(img1, img2, overlap)
            res = test_laplacian(top, bot, xy_offset)
            
            if res > best_result:
                best_result = res
                k2_offset = k1 - k2 + np.array([0,1])
    
    # 1 is bottom
    # first row
    for k1 in coords_1[coords_1[:, 1] == 0]:
        # last row
        for k2 in coords_2[coords_2[:, 1] == coords_2[:, 1].max()]:
    
            img1 = tile_map_1[tuple(k1)]
            img2 = tile_map_2[tuple(k2)]
    
            # img2 is top, img1 is bot
            xy_offset, top, bot = estimate_offset_vert(img2, img1, overlap)
            res = test_laplacian(top, bot, xy_offset)
            
            if res > best_result:
                best_result = res
                k2_offset = k1 - k2 - np.array([0,1])
    
    # 1 is right
    # first column
    for k1 in coords_1[coords_1[:, 0] == 0]:
        # last column
        for k2 in coords_2[coords_2[:, 0] == coords_2[:, 0].max()]:
    
            img1 = tile_map_1[tuple(k1)]
            img2 = tile_map_2[tuple(k2)]
            
            # img2 is left, img1 is right
            xy_offset, left, right = estimate_offset_horiz(img2, img1, overlap)
            res = test_laplacian(left, right, xy_offset)
            
            if res > best_result:
                best_result = res
                k2_offset = k1 - k2 - np.array([1,0])
    
    # 1 is left
    # last column
    for k1 in coords_1[coords_1[:, 0] == coords_1[:, 0].max()]:
        # first column
        for k2 in coords_2[coords_2[:, 0] == 0]:
    
            img1 = tile_map_1[tuple(k1)]
            img2 = tile_map_2[tuple(k2)]
    
            # img1 is left, img2 is right
            xy_offset, left, right = estimate_offset_horiz(img1, img2, overlap)
            res = test_laplacian(left, right, xy_offset)
            
            if res > best_result:
                best_result = res
                k2_offset = k1 - k2 + np.array([1,0])

    return k2_offset, best_result


def combine_stacks(pair, overlap):
    
    stack_1, stack_2 = pair
    pair_names = [s.stack_name for s in pair]

    # Get first common Z slice
    z = list(set(stack_1.slices).intersection(stack_2.slices))[0]
    
    z, tile_map_1, _ = load_tilemap({z: stack_1.slice_to_tilemap[z]}, stack_1.tile_maps_invert, True, True, 1)
    z, tile_map_2, _ = load_tilemap({z: stack_2.slice_to_tilemap[z]}, stack_2.tile_maps_invert, True, True, 1)

    k2_offset, _ = get_tile_maps_offset(tile_map_1, tile_map_2, overlap)

    # Combine tile_maps
    combined_tile_maps = {}
    for z in np.unique([stack_1.slices, stack_2.slices]).tolist():
        tile_map_1 = stack_1.slice_to_tilemap[z]
        tile_map_2 = stack_2.slice_to_tilemap[z]
        
        combined_tm = tile_map_1 | {tuple((np.array(k)+k2_offset).tolist()): v for k, v in tile_map_2.items()}
        min_index = np.array(list(combined_tm.keys())).min(0)
        combined_tm = {tuple((np.array(k)-min_index).tolist()):v for k, v in combined_tm.items()}
    
        combined_tile_maps[z] = combined_tm
    
    # Make combined stack
    combined_stack = Stack()
    combined_stack._set_tilemaps_paths(combined_tile_maps)
    combined_stack.tile_maps_invert = stack_1.tile_maps_invert | {tuple((np.array(k)+k2_offset).tolist()): v 
                                                                    for k, v in stack_2.tile_maps_invert.items()}
    
    combined_stack.stack_name = '-'.join(pair_names)
    
    return z, combined_stack, k2_offset