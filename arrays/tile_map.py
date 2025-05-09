import numpy as np

from .sift import estimate_transform_sift
from .utils import pad_to_shape
from ..io.tif import load_tilemap


def get_tile_map_margins(tile_space, margin, margin_boundaries=10):

    '''
    Compute margin per tile such that no data is cropped out at the boundaries of the image where no stitching is required.
    This ensures that no data is lost in case stitching between stacks is necessary.
    '''

    # top, bottom, left, right
    margin_overrides = {(x,y): [margin_boundaries]*4 for x in range(tile_space[1]) for y in range(tile_space[0])}

    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            margin_overrides[(x,y)][1] = margin
            margin_overrides[(x,y+1)][0] = margin

    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            margin_overrides[(x,y)][3] = margin
            margin_overrides[(x+1,y)][2] = margin

    return margin_overrides


def estimate_tiles_overlap(img1, 
                           img2, 
                           axis, 
                           scale):
    
    '''
    Estimate overlap between tiles supposed to be adjacent.
    '''
    
    overlap_search = max(img1.shape[axis], img2.shape[axis]) // 2
    
    if axis == 0:
        crop_img1 = img1[-overlap_search:, :]
        crop_img2 = img2[:overlap_search, :]
    elif axis == 1:
        crop_img1 = img1[:, -overlap_search:]
        crop_img2 = img2[:, :overlap_search]

    # First, try increasing only compute scale
    offset, _, valid_estimate = estimate_transform_sift(crop_img1, crop_img2, scale, refine_estimate=True)

    if valid_estimate:
        return overlap_search - np.abs(offset[::-1][axis])
    else:
        return 0


def estimate_tilemap_overlap(tile_map,
                             tile_space,
                             scale=0.1):
    
    '''
    Estimate the overlap between tiles of a tile_map. 
    '''
    
    overlaps_x = []
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            left = tile_map[(x,y)] 
            right = tile_map[(x+1,y)] 

            overlap = estimate_tiles_overlap(left, 
                                             right, 
                                             axis=1, 
                                             scale=scale)
            overlaps_x.append(overlap)

    overlaps_y = []
    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            bot = tile_map[(x,y)] 
            top = tile_map[(x,y+1)] 
            
            overlap = estimate_tiles_overlap(bot, 
                                             top, 
                                             axis=0, 
                                             scale=scale)
            overlaps_y.append(overlap)
    return int(np.max(overlaps_x + overlaps_y))


class TileMap:
    def __init__(self, 
                 z, 
                 tile_map_paths, 
                 tile_map=None, 
                 tile_masks=None, 
                 stack_name=None):
        self.z = z
        self.tile_map_paths = tile_map_paths
        self.stack_name = stack_name
        self.tile_map = tile_map
        self.tile_masks = tile_masks
        self.missing_tiles = []
        self.processing = {}

        if tile_map is not None:
            self.tile_space = (np.array(list(tile_map.keys()))[:,1].max()+1, 
                            np.array(list(tile_map.keys()))[:,0].max()+1)
            
        if self.tile_map is not None and self.tile_masks is None:
            self.tile_masks = {k: np.ones_like(v) for k,v in tile_map.items()}
            
    def _load_tile_map(self, processing=None):
        if processing is not None:
            self.processing = processing

        process_scheme = {
                        "gaussian": {"kernel_size": [3,3], "sigma": 1}, 
                        "clahe": {"clip_limit": 2, "tile_grid_size": [10,10]}
                        }
        process_scheme = {k:v for k,v in process_scheme.items() if self.processing.get(k)}
        _, self.tile_map, _ = load_tilemap({self.z: self.tile_map_paths}, 
                                            self.processing['tile_maps_invert'],
                                            process_scheme,
                                            self.processing['scale'],
                                            skip_missing=False)
        
    def homogenize_tile_shape(self):
        if len(self.tile_map) > 1:
            # There are more than one tiles
            # Pad tiles so they are all the same shape (required by sofima)
            max_shape = np.max([t.shape for t in self.tile_map.values()],axis=0)

            for k in self.tile_map: 
                tile = self.tile_map[k]
                mask = self.tile_masks[k]

                if np.any(np.array(tile.shape) != max_shape):
                    d = k[::-1] == (np.array(self.tile_space) - 1)
                    d[1] = np.logical_not(d[1])

                    tile = pad_to_shape(tile, max_shape, d.astype(int))
                    mask = pad_to_shape(mask, max_shape, d.astype(int))
                
                self.tile_map[k] = tile
                self.tile_masks[k] = mask
        
    def estimate_overlap(self, scale=0.1):
        if len(self.tile_map) > 1:
            self.overlap = estimate_tilemap_overlap(self.tile_map, self.tile_space, scale=scale)
            return self.overlap
        else:
            print('No overlap can be computed, only one image in tile_map.')
            self.overlap = None
            return None