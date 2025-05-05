import numpy as np


def xy_offset_to_pad(offset):
    pad = np.zeros([2,2], dtype=int)
    x,y = [int(i) for i in offset]
    
    if y > 0:
        pad[0][1] = y
    else:
        pad[0][0] = abs(y)
    
    if x > 0:
        pad[1][1] = x
    else:
        pad[1][0] = abs(x)

    return pad


def pad_to_shape(array, 
                 target_shape, 
                 direction=[1,1], 
                 pad_value=0):
    '''
    Pad an array to match a shape. If the target shape is smaller than the array's shape in a dimension, no padding is added to that dimension.
    '''
    assert array.ndim == len(target_shape)

    pad_size = target_shape-np.array(array.shape)
    pad_size[pad_size<0] = 0

    pad = np.zeros([2,2]).astype(int)
    for i, d in enumerate(direction):
        d = max(0, d)
        pad[i][d] = pad_size[i]
    return np.pad(array, pad, constant_values=pad_value)