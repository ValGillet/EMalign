
import numpy as np

from glob import glob
from collections import defaultdict

from .io import *


def parse_stack_info(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    tile_maps_paths = {}

    for z, tm in config['tile_maps'].items():
        tm = {tuple(int(i) 
                for i in re.findall(r'\b\d+\b', k)): v for k,v in tm.items()}
        tile_maps_paths.update({int(z): tm})

    tile_maps_invert = {tuple(int(i) for i in re.findall(r'\b\d+\b', k)): v 
                            for k,v in config['tile_maps_invert'].items()}
    return tile_maps_paths, tile_maps_invert


class Stack:
    def __init__(self, stack_path=None, stack_name=None, tile_maps_paths=None, tile_maps_invert=None):
        if stack_path is not None:
            self.stack_path = os.path.abspath(stack_path)
        else:
            self.stack_path = None
        
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
        tile_paths = glob(os.path.join(self.stack_path, '*.tif'))

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