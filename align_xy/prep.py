import json
import networkx as nx
import numpy as np
import os
import pandas as pd

from concurrent import futures
from itertools import combinations
from glob import glob
from tqdm import tqdm

from emprocess.utils.io import load_tif
from ..arrays.sift import estimate_transform_sift
from ..arrays.stacks import Stack
from ..visualize.nglancer import add_layers, start_nglancer_viewer
from ..align_z.utils import get_ordered_datasets


def find_offset_from_main_config(main_config_path):
    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    z_offsets = []
    for stack_config in main_config['stack_configs'].values():
        with open(stack_config, 'r') as f:
            stack_config = json.load(f)
        
        z_offsets.append(stack_config['z_end'])

    return max(z_offsets) + 1


def get_stacks(stack_paths, invert_instructions):

    '''
    Use a list of tileset paths and invert instructions to fetch stacks and split them into overlapping segments.
    Invert instructions are used to determine whether images in a tilest need to be inverted.
    '''

    # Load stacks
    stacks = []
    for stack_path in stack_paths:
        stack = Stack(stack_path)
        stack._get_tilemaps_paths()
        for k in stack.tile_maps_invert.keys():
            stack.tile_maps_invert[k]=invert_instructions[stack.stack_name]
        stacks.append(stack) 

    # Split stacks if there are overlaps
    unique_slices = sorted(np.unique(np.concatenate([stack.slices for stack in stacks])).tolist())
    df = pd.DataFrame({'z': unique_slices, 
                       'stack_name': [[] for _ in range(len(unique_slices))], 
                       'tile_paths':[[] for _ in range(len(unique_slices))]
                      })

    for stack in stacks:
        for z in stack.slices:
            # Join existing name and this stack at that slice
            df.loc[df.z == z, ['stack_name']] += [[stack.stack_name]]

            # Concatenate tile paths
            df.loc[df.z == z, ['tile_paths']] += [[stack.slice_to_tilemap[z]]]

    df['group'] = df['stack_name'].ne(df['stack_name'].shift()).cumsum()

    new_stacks = {}
    for group, group_df in df.groupby('group'):    
        stack_names = group_df.stack_name.iloc[0]

        if len(stack_names) == 1:
            # Stack name becomes name + group (gives an idea of order too)
            new_stack_name = str(group).zfill(2) + '_' + stack_names[0]

            tile_map = {}
            for z in group_df.z:
                tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[0]
            
            stack = Stack()
            stack.stack_name = new_stack_name
            stack._set_tilemaps_paths(tile_map)
            stack.tile_maps_invert = {k: invert_instructions[stack_names[0]] for k in tile_map[z].keys()}

            new_stacks[new_stack_name] = stack
        
        else:
            combined_stack_name = '_'.join([str(group).zfill(2)] + stack_names)
            pair = []
            for i in range(len(stack_names)):
                new_stack_name = str(group).zfill(2) + '_' + stack_names[i]
                
                tile_map = {}
                for z in group_df.z:
                    tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[i]

                stack = Stack()
                stack.stack_name = new_stack_name
                stack._set_tilemaps_paths(tile_map)
                stack.tile_maps_invert = {k: invert_instructions[stack_names[i]] for k in tile_map[z].keys()}

                pair.append(stack)
            new_stacks[combined_stack_name] = pair
        
    return new_stacks


def check_stacks_to_invert(stack_list, 
                           num_workers=1, 
                           **kwargs):

    '''Check what stacks must be inverted

    Display the first image of each stack in neuroglancer viewer, and prompt user to determine whether a stack needs to be inverted. 

    Args:
        stack_list (`list` of `emalign.align.xy.stacks.Stack`): List of stacks to visualize.
        num_workers (int, optional): Number of threads used to open images. Defaults to 1.
        **kwargs: Arguments passed to `start_nglancer_viewer`.

    Returns:
        to_invert (dict): Dictionary from stack_names to decision to invert: either True or False.
    '''

    viewer = start_nglancer_viewer(**kwargs)
    print('Neuroglancer viewer: ' + viewer.get_viewer_url())
    print('Please wait for images to load (CTRL+C to cancel).')

    to_invert = {}
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = {}
        for stack_path in sorted(stack_list):
            stack_name = stack_path.split('/')[-2]
            fs[stack_name] = tpe.submit(load_tif, 
                                        glob(stack_path + '*.tif')[0], 1, {})

        for i, (stack_name, f) in enumerate(fs.items()):
            arr = f.result()[0]
            add_layers([arr],
                        viewer,
                        names=[stack_name],
                        clear_viewer=True)
            
            answer = input(f'{str(i).zfill(2)}/{len(fs)} - Invert {stack_name}? (y/n) ').strip(' ')

            while answer not in ['y', 'n', '']:
                answer = input(f'{str(i).zfill(2)}/{len(fs)} - Please provide a valid answer for {stack_name}: (y/n) ')

            if answer == 'y' or answer == '':
                to_invert.update({stack_name: True})
            elif answer == 'n':
                to_invert.update({stack_name: False})
    return to_invert


# FUSE STACKS
def find_overlapping_stacks(dataset_paths):
    '''Find potentially overlapping stacks from a list of store paths. 

    Stacks are determined to be potentially overlapping if their z offsets match.
    
    Args:
        dataset_paths (`list` of `str`): List of absolute paths to zarr stores containing the image data.

    Returns:
        overlapping_stacks (`list` of `list`): List of lists containing groups of overlapping stacks.
    '''

    datasets, z_offsets = get_ordered_datasets(dataset_paths)
    datasets = datasets[::-1]
    z_offsets = z_offsets[::-1]

    overlapping_stacks = []
    while datasets:
        d = datasets.pop()
        z, z_offsets = z_offsets[-1], z_offsets[:-1]
        idx = np.where(z[0] == z_offsets[:,0])[0]
        z_offsets = np.delete(z_offsets, idx, axis=0)
        
        group = [d]
        if idx.size > 0:
            for i in idx[::-1]:
                group.append(datasets[i])
                del datasets[i]

        overlapping_stacks.append(group)

    return overlapping_stacks


def create_configs_fused_stacks(overlapping_stacks, 
                                scale=0.1):
    
    '''Create configurations for overlapping stacks.

    For stacks existing on the same Z levels, that were stitched "on grid" or images that could not, determine whether they do overlap and their transformations.

    Args:
        overlapping_stacks (list): List of stacks with the same Z offset, potentially overlapping on the XY plane.
        scale (`float`, optional): Scale to use to downsample images when determining offset with SIFT. Defaults to 0.1. 

    Returns:
        fuse_configs (`list` of `dict`): List of dictionaries of configuration of the stacks.\n
            Keys: Names of the stack found to be overlapping.\n
            path: Absolute path to the store containing the image data of this stack\n
            z_offset: Offset in pixel along the z axis.\n
            xy_offset: Offset (xy) in pixel to apply to the stack for images to be roughly aligned.\n
            rotation: Rotation (degrees) to be applied to the stack for images to be roughly aligned. 
    '''

    first_slices = {}
    overlap_G = nx.DiGraph()

    for dataset in overlapping_stacks:
        z = 0
        path = os.path.abspath(dataset.kvstore.path)
        stack_name = os.path.basename(path)
        img = dataset[z].read().result()
        while not img.any():
            # If latest slice before this dataset is empty, go to the previous one until finding a non-empty slice
            z += 1
            img = dataset[z].read().result()

        first_slices[stack_name] = (z, img)
        overlap_G.add_node(stack_name, path=path, z=z)

    for stack1, stack2 in tqdm(list(combinations(first_slices.keys(), 2))):
        img1 = first_slices[stack1][1]
        img2 = first_slices[stack2][1]
        offset, angle, valid_estimate = estimate_transform_sift(img1, img2, scale)

        if valid_estimate:
            overlap_G.add_edge(stack1, stack2, offset=-offset, angle=-angle)
            overlap_G.add_edge(stack2, stack1, offset=offset, angle=angle)

    fuse_configs = []
    for group in nx.connected_components(nx.to_undirected(overlap_G)):
        if len(group) == 1:
            continue

        abs_positions = {}
        abs_rotations = {}

        G = overlap_G.subgraph(group)
        start_node = list(group)[0]
        abs_positions[start_node] = np.array([0,0])
        abs_rotations[start_node] = 0

        for node in nx.bfs_tree(G, start_node):
            if node == start_node:
                continue

            path = nx.shortest_path(G, start_node, node)
        
            cumulative_rotation = 0
            cumulative_position = np.array([0,0])
            for i in range(len(path)-1):
                u,v = path[i], path[i+1]
                cumulative_rotation = (cumulative_rotation + G[u][v]['angle']) % 360
                cumulative_rotation = cumulative_rotation if cumulative_rotation<=180 else -180 + (cumulative_rotation % 180)
                cumulative_position = (cumulative_position + G[u][v]['offset'])
            abs_rotations[node] = cumulative_rotation
            abs_positions[node] = cumulative_position

        min_rotation = min(list(abs_rotations.values()))
        min_position = np.min(list(abs_positions.values()), axis=0)
        abs_rotations = {k:v-min_rotation for k,v in abs_rotations.items()}
        abs_positions = {k:v-min_position for k,v in abs_positions.items()}

        fuse_configs.append({
                    n: {'path': G.nodes[n]['path'],
                        'z_offset': G.nodes[n]['z'],
                        'xy_offset': -abs_positions[n],
                        'rotation': abs_rotations[n]} 
                        for n in G.nodes})
    return fuse_configs