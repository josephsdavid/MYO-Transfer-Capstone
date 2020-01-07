import numpy as np

# load up the first example first!

trial = np.fromfile("../PreTrainingDataset/Male0/training0/classe_0.dat", dtype = np.int16)
try2 = trial.reshape(int(trial.shape[0]/8),8)
abs((np.vstack(np.split(trial, int(trial.shape[0]/8))) - try2)).sum()

def read_file(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    return f.reshape(int(f.shape[0]/8),8)


def read_group_to_lists(path, n_classes = 7):
    # grab in all the male candidates
    for candidate in range(12):
        mens = [read_file(path + '/Male' + str(candidate) + '/training0/classe_%d.dat' %i) for i in range(n_classes*4)]

    # and all the female candidates
    for candidate in range(7):
        womens = [read_file(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' %i) for i in range(n_classes*4)]

    # combine and return!
    res = mens + womens
    return(res)

trials_all = read_group_to_lists("../PreTrainingDataset")


# longest run is 1034, so we will pad to that
maxlen = max([x.shape[0] for x in trials_all])

# padd around axis
# thankful for stack overflow today!
def pad_along_axis(array: np.ndarray, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return array
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b

trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]

[x.shape for x in trials_padded]

# get the list into an array with format (samples, timesteps, features)
# also update this to have labels
# finally, we probably want to use keras generators, because its faster and
# memory efficient, check out
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
