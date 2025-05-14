from collections import defaultdict
import networkx as nx
import numpy as np
import logging
from tqdm import tqdm
from itertools import combinations

from emalign.arrays.overlap import check_overlap
from emalign.arrays.sift import estimate_transform_sift
from emalign.arrays.stacks import Stack
from emalign.io.tif import load_tilemap


logging.basicConfig(level=logging.INFO)


def get_tile_positions_graph(G):

    '''
    Find positions of tiles in a graph.

    Args:

        G (``nx.DiGraph``):

            Fully connected directional graph containing tile keys as nodes, relative offset between tiles as edge attributes. 

    '''
    
    if not nx.is_connected(G):
        raise ValueError('Graph must be fully connected to determine tile positions')

    node_positions = {}

    node = list(G.nodes)[0]
    while len(G.nodes) != len(node_positions):
        if not node_positions:
            # If no tile has been processed yet, we assign this one as the reference
            node_positions[node] = np.array([0,0])

        for node in G.neighbors(node):
            edges = G.edges(node, data=True)

            # Iterate over the edges involving this node 
            # and assign a global offset to its neighbor
            for u, v, attrs in edges:
                rel_offset = attrs['rel_offset']

                if node == u:
                    if u not in node_positions or v in node_positions:
                        continue
                    node_positions[v] = (node_positions[u] - rel_offset).astype(int)
                elif node == v:
                    if v not in node_positions or u in node_positions:
                        continue
                    node_positions[u] = (node_positions[v] - rel_offset).astype(int)

    # Bring the smallest offset to (0,0) 
    min_position = np.min(np.stack(list(node_positions.values())), axis=0)
    for k,v in node_positions.items():
        node_positions[k] -= min_position

    tile_positions = defaultdict(dict)
    for key, new_pos in node_positions.items():
        stack_name, old_pos = key
        tile_positions[stack_name][old_pos] = tuple(new_pos.tolist())

    return tile_positions


def estimate_tile_map_positions(combined_stacks, 
                                apply_gaussian, 
                                apply_clahe, 
                                scale=[0.5, 1], 
                                overlap_score_threshold=0.8,
                                rotation_threshold=5):

    '''
    Given a list of overlaping image stacks, tries to calculate a transformation between each pair of tiles and check the overlap using a laplacian filter.
    Based on transformation, tiles are placed on a grid for further processing. Tiles that are found not to overlap well enough are split into multiple stacks.

    
    Args:

        combined_stacks (`list[Stack]`):

            List of overlapping image stacks.

        apply_gaussian (``bool``):

            Whether or not to apply gaussian filter with default parameters.

        apply_clahe (``bool``):

            Whether or not to apply CLAHE with default parameters.

        scale (`list[float]`):

            Scales to downsample images to for finding transformation with sift (1 = downsampling). 
            If offset computations fail at scale[0], will try scale[1].

        overlap_score_threshold (``float``):
         
            Determines the cutoff for how good overlap needs to be. Based on an index of overlap between 0 (bad) and 1 (perfect).

        rotation_threshold (``int``):

            Determines the maximum allowed rotation in degrees for a tile to be considered overlapping. 
            Too much rotation will mess with downstream computations for stitching. Will be implemented in the future.
    '''

    unique_slices, counts = np.unique([stack.slices for stack in combined_stacks], return_counts=True)
    z = int(unique_slices[counts == len(combined_stacks)][0])

    all_tiles = {}
    for stack in combined_stacks:
        z, tm, _ = load_tilemap({z: stack.slice_to_tilemap[z]}, stack.tile_maps_invert, apply_gaussian, apply_clahe, 1)

        for k,v in tm.items():
            all_tiles[(stack.stack_name, k)] = v 

    overlaps = []
    for k1, k2 in tqdm(list(combinations(all_tiles.keys(), 2)), position=0, desc='Estimating transformation between tiles...'):
        if k1[0] == k2[0]:
            # Same stack, different tiles, we know they overlap
            relative_offset = np.array(k1[1]) - np.array(k2[1])
            angle = 0
            overlap_score = 1
        else:
            # Different stacks, they may not overlap
            img1 = all_tiles[k1]
            img2 = all_tiles[k2]
            # ToDo: refactor this ugly thing
            try:
                offset, angle = estimate_transform_sift(img1, img2, scale[0])
            except:
                try:
                    offset, angle = estimate_transform_sift(img1, img2, scale[1])
                except:
                    offset = None
                    angle = 0

            if offset is not None:
                # Offset of k1 relative to k2
                relative_offset = np.abs(offset).argsort() * (offset/np.abs(offset)) * np.array([1,-1])
                overlap_score = check_overlap(img1, img2, 
                                                offset, angle, 
                                                threshold=overlap_score_threshold, 
                                                scale=scale,
                                                refine=True)
            else:
                relative_offset = (0,0)
                overlap_score = 0
        overlaps.append((k1, k2, relative_offset, angle, overlap_score))
        # u, v, relative xy_offset, angle, score

    # Create a graph connecting the different tilesets
    G = nx.Graph()
    for overlap in overlaps:
        # Offset of k1 relative to k2
        u, v, relative_offset, angle, overlap_score = overlap
        
        if overlap_score > overlap_score_threshold and angle < rotation_threshold:
            # Either the overlap score is good and we can guess position, or we know position because same tileset
            G.add_edge(u, v, rel_offset=relative_offset)
        
    G.add_nodes_from(list(all_tiles.keys()))

    logging.info('Figuring out tile positions')
    new_combined_stacks = []
    for subG in [G.subgraph(c) for c in nx.connected_components(G)]:
        tile_positions = get_tile_positions_graph(subG)

        remapped_tile_map = defaultdict(dict)
        remapped_tile_invert = {}
        for z in unique_slices:
            for stack in combined_stacks:
                if stack.stack_name not in tile_positions:
                    continue
                for old_pos, new_pos in tile_positions[stack.stack_name].items():
                    remapped_tile_map[int(z)][new_pos] = stack.slice_to_tilemap[z][old_pos]
                    # No need to assign tile invert for every slice, but shorter and quick
                    remapped_tile_invert[new_pos] = stack.tile_maps_invert[old_pos]            

        names = np.unique([n[0] for n in subG.nodes])
        index = names[0].split('_')[0]

        combined_stack = Stack()
        combined_stack.stack_name = '_'.join([index] + [n.split('_', maxsplit=1)[-1] for n in names])
        combined_stack._set_tilemaps_paths(remapped_tile_map)
        combined_stack.tile_maps_invert = remapped_tile_invert

        new_combined_stacks.append(combined_stack)

    assert sum([len(s.tile_maps_invert.keys()) for s in new_combined_stacks]) == sum([len(s.tile_maps_invert.keys()) for s in combined_stacks])

    return new_combined_stacks