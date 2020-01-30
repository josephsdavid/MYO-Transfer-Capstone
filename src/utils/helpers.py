import numpy as np

def read_file_validation(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    out = f.reshape(int(f.shape[0]/8),8)
    return out.astype(np.float16)


def pad_along_axis(array: np.ndarray, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return array[:target_length,:]
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b
