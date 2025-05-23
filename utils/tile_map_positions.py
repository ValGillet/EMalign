from collections import defaultdict
import networkx as nx
import numpy as np
import cv2
import logging
from tqdm import tqdm
from itertools import combinations

from emprocess.utils.transform import rotate_image
from emalign.utils.io import load_tilemap
from emalign.utils.stacks import Stack
from emalign.utils.offsets import estimate_transform_sift, xy_offset_to_pad
from emalign.utils.arrays import pad_to_shape

logging.basicConfig(level=logging.INFO)


def get_overlap(img1, img2, xy_offset, rotation_angle):

    '''
    Extract overlapping parts of two images based on an offset and rotation from img2 to img1.
    '''

    # Estimate overlap
    # Masks and images are padded to same shape to facilitate comparison
    # I'm sure there is a smartest way but this works
    mask1 = np.ones_like(img1)
    mask1 = rotate_image(img1, -rotation_angle)
    img1 = rotate_image(img1, -rotation_angle)

    mask2 = np.ones_like(img2).astype(bool)
    img1=np.pad(img1, xy_offset_to_pad(-xy_offset))
    img2=np.pad(img2, xy_offset_to_pad(xy_offset))

    mask1=np.pad(mask1, xy_offset_to_pad(-xy_offset))
    mask2=np.pad(mask2, xy_offset_to_pad(xy_offset))

    max_shape = np.max([img1.shape, img2.shape], axis=0)

    img1 = pad_to_shape(img1, max_shape)
    img2 = pad_to_shape(img2, max_shape)
    mask1 = pad_to_shape(mask1, max_shape)
    mask2 = pad_to_shape(mask2, max_shape)
    
    mask = mask1.astype(bool) & mask2.astype(bool)

    if mask.any():
        y1,x1 = np.min(np.where(mask), axis=1) 
        y2,x2 = np.max(np.where(mask), axis=1) 

        return img1[y1:y2, x1:x2], img2[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    else:
        return None
    

def compute_laplacian_var_diff(overlap_1, overlap_2, mask):

    '''
    Compute a metric ([0,1]) describing how well two arrays overlap, based on laplacian filter.
    If score is 1, overlapping regions have the same edge content and therefore overlap well.
    '''
    
    mask = np.ones_like(overlap_1).astype(bool) if mask is None else mask

    laplacian1 = cv2.Laplacian(overlap_1, cv2.CV_64F)[mask]
    laplacian2 = cv2.Laplacian(overlap_2, cv2.CV_64F)[mask]

    lap_var1 = np.var(laplacian1)
    lap_var2 = np.var(laplacian2)

    # Calculate an index of difference in edge content (variance of laplacian)
    # Between 0 and 1, low means exact same content, 1 means different
    return 1 - abs(lap_var1 - lap_var2) / max(lap_var1, lap_var2)


def check_overlap(img1, img2, xy_offset, theta, threshold=0.5, scale=(0.3, 0.5), refine=True):

    '''
    Compute a metric describing how well images overlap, based on a given offset and rotation. 
    '''

    # Index of sharpness using Laplacian
    overlap = get_overlap(img1, img2, xy_offset, theta)

    if overlap is not None:
        overlap1, overlap2, mask = overlap

        lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)

        if refine and lap_variance_diff < threshold:
            logging.debug('Refining overlap estimation...')
            # Retry the overlap, it can often get better
            try:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[0])
            except:
                xy_offset, theta = estimate_transform_sift(overlap1, overlap2, scale=scale[1])
            res = get_overlap(overlap1, overlap2, xy_offset, theta)
            
            if res is not None:
                overlap1, overlap2, mask = res
                lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask)
            else:
                lap_variance_diff = 0
    else:
        # Images do not overlap (displacement is larger than image itself)
        lap_variance_diff = 0

    return lap_variance_diff


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