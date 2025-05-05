''' Utilities for alignment of tilesets in XY plane.'''

import cv2
import numpy as np
import os
import re
import warnings

from concurrent import futures
from glob import glob
from tqdm import tqdm

from ..array.utils import compute_laplacian_var_diff, get_overlap


def get_tileset_resolution(tileset_path):
    '''
    Find resolution of a stack by reading an .info file located in the stack directory.
    '''
    info=None
    with os.scandir(tileset_path) as entries:
        for entry in entries:
            if entry.name.endswith('.info'):
                info = entry.path
                break
    
    if info is None:
        return None
    
    with open(info, 'r') as f:
        content = f.readlines()

    resolution = tuple(map(int, re.findall(r'\d+', content[5])))

    return (tileset_path, resolution)


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    
    # Get all directories containing tilesets that are present in main_dir    
    tileset_dirs = glob(main_dir + '/*/')

    stack_list = []
    # Find the ones with the right resolution
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = []
        for d in tileset_dirs:
            fs.append(tpe.submit(get_tileset_resolution, d))

        for f in tqdm(futures.as_completed(fs), total=len(fs), desc=f'Looking for resolution: {resolution}', leave=False):
            result = f.result()
            if result is None:
                continue
            # Find the directory with the right pattern if relevant
            for d in dir_pattern:
                if d in result[0].split('/')[-2] and result[1] == tuple(resolution):
                    stack_list.append(result[0])
    return sorted(stack_list)


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
