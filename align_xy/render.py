from sofima import warp

from .utils import check_stitch
from ..io.store import write_slice


def render_slice_xy(destination,
                    z,
                    tile_map,
                    meshes,
                    stride,
                    tile_masks=None,
                    parallelism=1,
                    margin=50,
                    dest_mask=None,
                    return_render=False,
                    **kwargs):

    if len(tile_map) > 1:
        # Render stitched image
        stitched, mask, warped_tiles = warp.render_tiles(tile_map, meshes, 
                                                    tile_masks=tile_masks, 
                                                    parallelism=parallelism, 
                                                    stride=(stride, stride), 
                                                    return_warped_tiles=True,
                                                    margin=margin,
                                                    **kwargs)
        # Evaluate overlap
        stitch_score = check_stitch(warped_tiles, margin)
    else:
        stitched = list(tile_map.values())[0]
        stitch_score = 1
    
    if return_render:
        return stitched, stitch_score
    else:
        write_slice(destination, stitched, z)

        if dest_mask is not None:
            write_slice(dest_mask, mask, z)

        return stitch_score