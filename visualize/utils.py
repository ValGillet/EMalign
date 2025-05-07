import numpy as np

def assemble_tile_map(tile_map):

    max_shape = np.max([t.shape for t in tile_map.values()], axis=0)
    max_coords = np.max([c for c in tile_map.keys()], axis=0)[::-1]
    
    test_combined = np.zeros((max_coords+1)*max_shape) 
    
    for coords, tile in tile_map.items():
        origin = np.array(coords)[::-1]*max_shape
        end = origin + np.array(tile.shape)
        test_combined[origin[0]:end[0], origin[1]:end[1]]=tile

    return test_combined.astype(np.uint8)