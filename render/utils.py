''' Utilities for checking aligned images.'''


import cv2
import numpy as np
import warnings

from emalign.array.utils import compute_laplacian_var_diff, get_overlap


def check_stitch(warped_tiles, margin):

    tile_space = (np.array(list(warped_tiles.keys()))[:,1].max()+1, 
                  np.array(list(warped_tiles.keys()))[:,0].max()+1)
    
    overlap_scores = []
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            x1,y1,left = warped_tiles[(x,y)] 
            x2,y2,right = warped_tiles[(x+1,y)]

            offset = np.array([x1-x2, y1-y2])
            
            overlap = get_overlap(left, right, offset, 0)

            if overlap is None:
                overlap_score = 0
            else:
                overlap1, overlap2, _ = overlap
                try:
                    overlap_score = compute_laplacian_var_diff(overlap1[:, :-margin], 
                                                               overlap2[:, margin:],
                                                               None)
                except cv2.error as e:
                    if e.err == '!_src.empty()':
                        overlap_score = 0
                        warnings.warn('Empty overlap. There may not be overlap between tiles. Overlap score set to 0.')
                    else:
                        raise e
            overlap_scores.append(overlap_score)

    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            x1,y1,bot = warped_tiles[(x,y)] 
            x2,y2,top = warped_tiles[(x,y+1)] 
            
            offset = np.array([x1-x2, y1-y2])
            
            overlap = get_overlap(bot, top, offset, 0)

            if overlap is None:
                overlap_score = 0
            else:
                overlap1, overlap2, _ = overlap
                try:
                    overlap_score = compute_laplacian_var_diff(overlap1[:-margin, :], 
                                                               overlap2[margin:, :],
                                                               None)
                except cv2.error as e:
                    if e.err == '!_src.empty()':
                        overlap_score = 0
                        warnings.warn('Empty overlap. There may not be overlap between tiles. Overlap score set to 0.')
                    else:
                        raise e
            overlap_scores.append(overlap_score)
    return overlap_scores
