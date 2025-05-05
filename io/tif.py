from emprocess.utils.io import load_tif


def load_tilemap(tile_map_paths, invert, process_scheme, scale, skip_missing=False):
    
    '''
    Load a tile map based on provided paths. Apply image processing if specified.
    '''

    z, tile_map_paths= list(tile_map_paths.items())[0]

    if not isinstance(invert, dict):
        invert = dict(zip(list(tile_map_paths.keys()), [invert]*len(tile_map_paths)))
    
    tile_map = {}
    tile_map_ds = {}
    for yx_pos, tile_path in tile_map_paths.items():
        if invert[yx_pos]:
            proc = process_scheme.copy()
            proc['invert'] = True
        else:
            proc = {k:v for k,v in process_scheme.items() if k != 'invert'}

        try:
            img, img_ds, _ = load_tif(tile_path, scale=scale, process_scheme=proc, compute_mask=False)
        except Exception as e:
            if skip_missing:
                img, img_ds = None, None
            else:
                raise e
        tile_map[yx_pos] = img
        tile_map_ds[yx_pos] = img_ds

    return z, tile_map, tile_map_ds