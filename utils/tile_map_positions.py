from collections import defaultdict
import networkx as nx
import numpy as np
import cv2
import logging
from tqdm import tqdm

from emprocess.utils.img_proc import downsample
from emalign.utils.io import load_tilemap

logging.basicConfig(level=logging.INFO)


# TODO:
# Figure out amount of overlap between tiles from pixel offset


def estimate_transform_sift(img1, img2, scale=1):

    ds_img1 = downsample(img1, scale)
    ds_img2 = downsample(img2, scale)

    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ds_img1,None)
    kp2, des2 = sift.detectAndCompute(ds_img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if des1 is None or des2 is None:
        return None, None, 0
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Estimate affine transformation matrix
    try:
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    except:
        return None, None, 0    
    if M is None:
        return None, None, 0
    
    # # Extract translation offsets
    xy_offset = M[:, 2]

    # # Extract rotation angle in degrees
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    # xy offset is from img2 to img1
    return xy_offset/scale, theta, mask.sum()


def get_tile_positions_connected_graph(G):
    node_positions = {}

    while True:
        # Not sure if this is necessary, but there might be a possibility to miss a node if it was skipped
        # I cannot bother checking whether that's truly possible or not
        for node in G:
            if len(node_positions) == 0:
                # If no tile has been processed yet, we assign this one as the reference
                node_positions[node] = np.array([0,0])

            edges = [(u,v,d) for u,v,d in G.edges(data=True) if node in (u,v)]

            if len(edges) == 0:
                # All its neighbors were processed
                continue

            # Iterate over the edges involving this node 
            # and assign a global offset to its neighbor
            done = []
            for edge in edges:
                u, v, attrs = edge
                rel_offset = attrs['rel_offset']

                if node == u:
                    if u not in node_positions or v in node_positions:
                        continue
                    node_positions[v] = (node_positions[u] + rel_offset).astype(int)
                    done.append(True)
                elif node == v:
                    if v not in node_positions or u in node_positions:
                        continue
                    node_positions[u] = (node_positions[v] + rel_offset).astype(int)
                    done.append(True)
        if np.all([node in node_positions for node in G]):
            break
        else:
            logging.DEBUG('Going through graph again')

    # Bring the smallest offset to (0,0) 
    min_position = np.min(np.stack(list(node_positions.values())), axis=0)
    for k,v in node_positions.items():
        node_positions[k] -= min_position

    tile_positions = defaultdict(dict)
    for key, new_pos in node_positions.items():
        stack_name, old_pos = key
        tile_positions[stack_name][old_pos] = tuple(new_pos.tolist())

    return tile_positions


def estimate_tile_map_positions(combined_stacks, apply_gaussian, apply_clahe, scale=0.3):
    unique_slices, counts = np.unique([stack.slices for stack in combined_stacks], return_counts=True)
    z = int(unique_slices[counts == len(combined_stacks)][0])

    all_tiles = {}
    for stack in combined_stacks:
        z, tm, _ = load_tilemap({z: stack.slice_to_tilemap[z]}, stack.tile_maps_invert, apply_gaussian, apply_clahe, 1)

        for k,v in tm.items():
            all_tiles[(stack.stack_name, k)] = v 

    overlaps = []
    for k1 in tqdm(all_tiles.keys(), position=0, desc='Estimating transformation between tiles...'):
        for k2 in all_tiles.keys():
            if k1 == k2:
                continue
            overlaps.append((k2, k1, *estimate_transform_sift(all_tiles[k1], all_tiles[k2], scale)))
            # u, v, yx_offset, angle, score

    G = nx.Graph()
    score_threshold = 20

    for overlap in overlaps:
        # Offset of k1 relative to k2
        u, v, offset, angle, score = overlap

        if offset is None:
            continue
        offset = -offset
        relative_offset = np.abs(offset).argsort() * (offset/np.abs(offset)) * np.array([1,-1])

        if score > score_threshold:
            G.add_edge(u, v, offset=offset, rel_offset=relative_offset)
        
    G.add_nodes_from(list(all_tiles.keys()))
    if nx.is_connected(G):
        # All tiles are connected and overlapping
        logging.info('Figuring out tile positions')
        tile_positions = get_tile_positions_connected_graph(G)

    remapped_tile_map = defaultdict(dict)

    for count, z in zip(counts, unique_slices):
        for stack in combined_stacks:
            for old_pos, new_pos in tile_positions[stack.stack_name].items():
                remapped_tile_map[int(z)][new_pos] = stack.slice_to_tilemap[z][old_pos]

    remapped_tile_invert = {}

    for stack in combined_stacks:
        for old_pos, new_pos in tile_positions[stack.stack_name].items():
            remapped_tile_invert[new_pos] = stack.tile_maps_invert[old_pos]

    return remapped_tile_map, remapped_tile_invert