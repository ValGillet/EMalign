import cv2 as cv
import functools as ft
import jax
import jax.numpy as jnp
import numpy as np

from PIL import Image
from sofima import stitch_rigid, stitch_elastic, mesh, flow_utils

from .offsets import *


def test_laplacian(padded_img1, padded_img2, xy_offset):
    pad = xy_offset_to_pad(xy_offset)

    padded_img1 = np.pad(padded_img1, pad)
    padded_img2 = np.pad(padded_img2, pad[:, ::-1])

    x,y = [abs(int(n)) for n in xy_offset]
    mask = (padded_img1>0) & (padded_img2>0)
    overlap = np.mean([padded_img1*mask, padded_img2*mask], axis=0)[y:-y, x:-x]
    
    lp = cv.Laplacian(np.round(overlap).astype(np.uint8), cv.CV_16S)
    
    return cv.convertScaleAbs(lp).sum()


def rescale_mesh(mesh, scale):

    '''
    Upscale mesh. Used to upsample meshes that were computed with a downsample version of an image.
    '''
   
    m1, m2 = mesh*scale
    m1 = Image.fromarray(m1.squeeze())
    m2 = Image.fromarray(m2.squeeze())

    m1 = np.array(m1.resize((m1.width*scale, m1.height*scale), resample=Image.Resampling(0)))
    m2 = np.array(m2.resize((m2.width*scale, m2.height*scale), resample=Image.Resampling(0)))

    return np.stack([m1[None, :,: ],m2[None, :, :]])


def get_coarse_offset(tile_map, 
                      tile_space,
                      overlap=0.15,
                      filter_size=5):
    '''
    Compute coarse offset and mesh for initial rigid XY alignment
    '''
    
    if overlap < 1:
        overlap = np.round(np.array(tile_map[0,0].shape) * overlap).astype(int)
    else:
        overlap = (overlap, overlap)

    # Coarse rigid offset between tiles
    cx, cy = stitch_rigid.compute_coarse_offsets(tile_space, tile_map, 
                                                 overlaps_xy=((10,overlap[1]),(10,overlap[0])),
                                                 filter_size=filter_size)
    
    # Figure out offset from neighbors if missing
    cx = stitch_rigid.interpolate_missing_offsets(cx, -1)
    cy = stitch_rigid.interpolate_missing_offsets(cy, -2)
    cx[np.isinf(cx)] = np.nan
    cy[np.isinf(cy)] = np.nan

    coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy)

    return cx, cy, coarse_mesh
            


def get_elastic_mesh(tile_map, 
                     cx, 
                     cy, 
                     coarse_mesh, 
                     stride=20):
    
    ''' 
    Compute elastic mesh for XY alignment.
    '''

    # Elastic alignment
    cx = cx[:,0,...]
    cy = cy[:,0,...]
    fine_x, offsets_x = stitch_elastic.compute_flow_map(tile_map, 
                                                        cx, 
                                                        0, 
                                                        stride=(stride, stride),
                                                        batch_size=4)
    fine_y, offsets_y = stitch_elastic.compute_flow_map(tile_map, 
                                                        cy, 
                                                        1,
                                                        stride=(stride, stride),
                                                        batch_size=4)
    
    kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, "max_deviation": 5, "max_magnitude": 0}
    fine_x = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
    fine_y = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}
    
    kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
    fine_x = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
    fine_y = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}
    
    data_x = (cx, fine_x, offsets_x)
    data_y = (cy, fine_y, offsets_y)
    
    fx, fy, x, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
        data_x, data_y, list(tile_map.keys()),
        coarse_mesh[:, 0, ...], stride=(stride, stride),
        tile_shape=next(iter(tile_map.values())).shape)
    
    @jax.jit
    def prev_fn(x):
      target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=x, fx=fx,
                             fy=fy, stride=(stride, stride))
      x = jax.vmap(target_fn)(nbors)
      return jnp.transpose(x, [1, 0, 2, 3])
    
    # These detault settings are expect to work well in most configurations. Perhaps
    # the most salient parameter is the elasticity ratio k0 / k. The larger it gets,
    # the more the tiles will be allowed to deform to match their neighbors (in which
    # case you might want use aggressive flow filtering to ensure that there are no
    # inaccurate flow vectors). Lower ratios will reduce deformation, which, depending
    # on the initial state of the tiles, might result in visible seams.
    k0 = 0.01 # inter-section springs (elasticity)
    # High k0 results in images that tend to "fold" onto themselves

    k = 0.1 # intra-section springs (elasticity)
    gamma = 0 # dampening factor

    config = mesh.IntegrationConfig(dt=0.001, gamma=gamma, k0=k0, k=k, stride=(stride, stride),
                                    num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                    dt_max=100, prefer_orig_order=True,
                                    start_cap=0.1, final_cap=10., remove_drift=True)
    
    x, _, _ = mesh.relax_mesh(x, None, config, prev_fn=prev_fn)
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    meshes = {idx_to_key[i]: np.array(x[:, i:i+1 :, :]) for i in range(x.shape[1])}
    
    return meshes



def compute_coarse_offset_sift(tile_space, tile_map, overlap):
    conn_x = np.full((2, 1, tile_space[0], tile_space[1]), np.nan)
    axis = 0
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            if not ((x, y) in tile_map and (x + 1, y) in tile_map):
                continue

            left = tile_map[(x, y)]
            right = tile_map[(x + 1, y)]

            offset = estimate_transform_sift(right[:, :overlap[1]], left[:, -overlap[1]:], 0.5)[0]

            if offset is None:
                offset = [np.inf, np.inf]
            else:
                offset[axis] -= overlap[1]

            conn_x[:, 0, y, x] = offset

    conn_y = np.full((2, 1, tile_space[0], tile_space[1]), np.nan)
    axis = 1
    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            if not ((x, y) in tile_map and (x, y + 1) in tile_map):
                continue

            top = tile_map[(x, y)]
            bot = tile_map[(x, y + 1)]

            offset = estimate_transform_sift(bot[:overlap[0], :], top[-overlap[0]:, :], 0.5)[0]
            
            if offset is None:
                offset = [np.inf, np.inf]
            else:
                offset[axis] -= overlap[0]

            conn_y[:, 0, y, x] = offset


