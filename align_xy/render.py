import numpy as np

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
    '''Render an aligned image from a tile map.

    Use a tile_map and corresponding meshes to produce an aligned image and mask. 
    Overlaps are assessed with check_stitch to produce a stitch_score that will be logged to find flawed slices.
    The score is based on a laplacian filter, between 0 (no similarity) and 1 (exact match).

    Args:
        destination (`tensorstore.TensorStore`): Zarr store where to write aligned slice.
        z (int): Z index at which to write the slice (axis at first position).
        tile_map (dict of `np.ndarray`): Dictionary from [x,y] tile position to [y,x] image.
        meshes (dict of `np.ndarray`): Dictionary from [x,y] tile position to [2, z, x, y] array of mesh positions. Order of keys determines the order of render.
        stride (int): Step used to determine mesh node positions.
        tile_masks (dict of `np.ndarray`, optional): Dictionary from [x,y] tile position to [y,x] boolean masks corresponding to tile_map. Defaults to None.
        parallelism (int, optional): Number of threads used by warp.render_tiles to warp tiles in parallel (max one thread per tile). Defaults to 1.
        margin (int, optional): Number of pixels cropped from each tile's boundaries to remove artifacts from deformation. Defaults to 50.
        dest_mask (_type_, optional): Zarr store where to write aligned slice's mask. Defaults to None.
        return_render (bool, optional): Whether to return the aligned image rather than writing it. Defaults to False.
        **kwargs (optional): Additional arguments passed to warp.render_tiles. 
            e.g.: margin_overrides provides specific margins per direction per tile.

    Returns:
        int: 
            If return_render == False: stitch score describing how well overlaps match, between 0 and 1 as defined by check_stitch. 
            If return_render == True: tuple of: aligned image, stitch score.
    '''

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
        mask = np.ones_like(list(tile_masks.values())[0]).astype(bool)
        stitch_score = 1
    
    if return_render:
        return stitched, stitch_score
    else:
        write_slice(destination, stitched, z)

        if dest_mask is not None:
            write_slice(dest_mask, mask, z)

        return stitch_score