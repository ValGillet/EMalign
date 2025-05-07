'''
Functions related to neuroglancer viewer.
'''
import numpy as np


def start_nglancer_viewer(bind_address='localhost',
                          bind_port=33333):
    import neuroglancer

    neuroglancer.set_server_bind_address(bind_address=bind_address, bind_port=bind_port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layout = 'xy'

    return viewer


def data_to_LocalVolume(
                      data,
                      spatial_dims,
                      voxel_offset,
                      voxel_size,
                      vtype
                       ):
    
    '''
    Based on funlib implementation
    '''
    import neuroglancer

    if data.dtype == bool:
        data = data.astype(np.uint8)

    spatial_dim_names = ['t', 'x', 'y', 'z']
    channel_dim_names = ['b^', 'c^']

    dims = len(data.data.shape)
    channel_dims = dims - spatial_dims
    voxel_offset = [0] * channel_dims + list(voxel_offset)

    attrs = {
             'names': (channel_dim_names[-channel_dims:] if channel_dims > 0 else [])
             + spatial_dim_names[-spatial_dims:],
             'units': [''] * channel_dims + ['nm'] * spatial_dims,
             'scales': [1] * channel_dims + list(voxel_size),
            }

    dimensions = neuroglancer.CoordinateSpace(**attrs)
    local_volume = neuroglancer.LocalVolume(
                                            data=data,
                                            voxel_offset=voxel_offset,
                                            dimensions=dimensions,
                                            volume_type=vtype
                                           )
    return local_volume


def add_layers(arrays,
               viewer,
               names=[],
               voxel_offsets=[],
               voxel_sizes=[],
               vtypes=[],
               clear_viewer=True):
    
    if not names:
        names = list(map(str, range(len(arrays))))
    if not voxel_offsets:
        voxel_offsets = [[0,0,0]]*len(arrays)
    if not voxel_sizes:
        voxel_sizes = [[1,1,1]]*len(arrays)
    if not vtypes:
        vtypes = [None]*len(arrays)

    layers = {}
    for i, arr in enumerate(arrays):
        name = names[i]
        voxel_offset = voxel_offsets[i]
        voxel_size = voxel_sizes[i]
        vtype = vtypes[i]


        if vtype is not None:
            assert vtype in ['segmentation', 'image']
        elif arr.dtype == np.uint8:
            if arr.max() == 1:
                # Mask
                vtype = 'segmentation'
            else:
                # Image
                vtype = 'image'
        elif arr.dtype == np.uint64 or arr.dtype == bool:
            vtype = 'segmentation'
        else:
            vtype = 'image'

        arr = arr[None, ...] if arr.ndim == 2 else arr
        layers[name] = data_to_LocalVolume(arr, 
                                           len(arr.shape),
                                           voxel_offset,
                                           voxel_size,
                                           vtype
                                           )

    if clear_viewer:
        with viewer.txn() as s:
            s.layers.clear()
    
    with viewer.txn() as s:
        for name, layer in layers.items():
            s.layers.append(name=name, layer=layer)