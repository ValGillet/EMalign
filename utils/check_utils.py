import numpy as np
import sys

from concurrent import futures
from glob import glob
from PIL import ImageDraw,ImageFont

from .io_utils import *
from .stacks_utils import *


def assemble_tile_map(tile_map):

    max_shape = np.max([t.shape for t in tile_map.values()], axis=0)
    max_coords = np.max([c for c in tile_map.keys()], axis=0)[::-1]
    
    test_combined = np.zeros((max_coords+1)*max_shape) 
    
    for coords, tile in tile_map.items():
        origin = np.array(coords)[::-1]*max_shape
        end = origin + np.array(tile.shape)
        test_combined[origin[0]:end[0], origin[1]:end[1]]=tile

    return test_combined.astype(np.uint8)


def check_stacks_to_invert(stack_list, resolution, num_workers, port=33333):
    import neuroglancer 
    stacks_to_check = {}
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = []
        for stack_path in stack_list:
            fs.append(tpe.submit(load_tif, 
                                 glob(stack_path + '*.tif')[0], 
                                 False, 
                                 True, 
                                 True, 
                                 0.5))

        for stack_path, f in tqdm(zip(stack_list, fs), desc='Loading example tifs', total=len(fs), leave=False):
            _, arr = f.result()
            stacks_to_check.update({stack_path.split('/')[-2]: arr})

    # VIEWER
    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=port)

    dimensions = neuroglancer.CoordinateSpace(names=['y', 'x'], units='nm', scales=resolution)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.dimensions = dimensions
        row_1 = []
        row_2 = []
        for i, (stack_name, img) in enumerate(stacks_to_check.items()):
            s.layers[stack_name] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(img.T, dimensions)) # Data has same ndim as specified in dimensions
            if i%2:
                row_2.append(neuroglancer.LayerGroupViewer(layers=[stack_name], layout = 'xy'))
            else:
                row_1.append(neuroglancer.LayerGroupViewer(layers=[stack_name], layout = 'xy'))

        s.layout = neuroglancer.column_layout([neuroglancer.row_layout(row_1), neuroglancer.row_layout(row_2)])
    
    url = viewer.get_viewer_url()
    print(f'Remote Viewer: {url}')
    print('Local viewer: ' + 'http://localhost:' + url.split(':')[-1])
    
    to_invert = {}
    for stack_name in stacks_to_check.keys():
        answer = input(f'Invert {stack_name}? (y/n) ')
        if answer == 'y':
            to_invert.update({stack_name: True})
        elif answer == 'n':
            to_invert.update({stack_name: False})
    return to_invert


def check_tile_map_positions(tile_map, resolution, port):
    import neuroglancer
    
    # Write tile position on image
    tile_map_text = {}
    for k, tile in tile_map.items():
        img = Image.fromarray(tile)
        ImageDraw.Draw(img).text((100,100), 
                                align='right',
                                text=str(k), font=ImageFont.truetype('FreeMono.ttf', 200))
        
        tile_map_text[k] = np.array(img)
    assembled_tile_map = assemble_tile_map(tile_map_text).astype(np.uint8)
    
    # VIEWER
    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=port)

    dimensions = neuroglancer.CoordinateSpace(names=['x', 'y'], units='nm', scales=resolution)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.dimensions = dimensions
        s.layers['tile_map'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(assembled_tile_map.T, dimensions))

    url = viewer.get_viewer_url()
    print(f'Remote Viewer: {url}')
    print('Local viewer: ' + 'http://localhost:' + url.split(':')[-1])
    
    return input('Does this combined stack look good? (y/n) ')


def check_combined_stacks(stack_1, 
                          stack_2, 
                          overlap, 
                          apply_gaussian, 
                          apply_clahe, 
                          scale,
                          resolution,
                          port):
    
    # Detect overlapping regions
    z, combined_stack, _ = combine_stacks([stack_1, stack_2], overlap)
    z, _, tile_map_ds = load_tilemap({z: combined_stack.slice_to_tilemap[z]}, 
                                     combined_stack.tile_maps_invert, 
                                     apply_gaussian, 
                                     apply_clahe, 
                                     scale)

    while check_tile_map_positions(tile_map_ds, resolution, port) != 'y':
        new_keys = {}
        for k in tile_map_ds.keys():
            new_k = input(f'Propose new position (x y) for {k}: ')
            new_keys.update({k: tuple(map(int, new_k.split(' ')))})

        tile_map_ds = {new_keys[k]: tm for k,tm in tile_map_ds.items()}

        new_tile_map = defaultdict(dict)
        for z, tile_map in combined_stack.slice_to_tilemap.items():
            new_tile_map.update({z: {new_keys[k]: tm for k,tm in tile_map.items()}})
        
        new_tile_map_invert = {new_keys[k]: combined_stack.tile_maps_invert[k] for k in tile_map.keys()}
        combined_stack._set_tilemaps_paths(new_tile_map)
        combined_stack.tile_maps_invert = new_tile_map_invert
        print('Check new solution: ')
    
    return combined_stack
