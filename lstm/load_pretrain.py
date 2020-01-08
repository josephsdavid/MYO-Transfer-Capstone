import numpy as np
from sklearn.preprocessing import MinMaxScaler

# load up the first example first!
def read_file(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    return f.reshape(int(f.shape[0]/8),8)


def read_group_to_lists(path, n_classes = 7):
    # grab in all the male candidates
    res = []
    labels = []
    trials = range(n_classes*4)
    for candidate in range(12):
        man = [read_file(path + '/Male' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
        # list addition is my new favorite python thing
        labs = [t % n_classes for t in trials]
        res += man
        labels += labs

    # and all the female candidates
    for candidate in range(7):
        woman = [read_file(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
        labs = [t % n_classes for t in trials]
        res += woman
        labels += labs

    return res, labels

# longest run is 1038, so we will pad to that
# pad around axis
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

def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.dstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )






# window size = 260 ms, as per the original paper
# data collected at 200 hz, so roughly 5 seconds of data
# stepsize = 5 seconds, as per original paper
# trials_rolled = [window_stack(x, 5, int(260/5)) for x in trials_padded]

def roll_labels(x, y):
    labs_rolled = []
    for i in range(len(y)):
        l = y[i]
        n = x[i].shape[2]
        labs_rolled.append(np.repeat(l,n))
    return np.hstack(labs_rolled)


def read_data(path, n_classes = 7, scale = False):
    # read in trials and label them
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    trials_rolled = [window_stack(x, 21, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

# for use with TimeSeries Generator
def read_data_unrolled(path, n_classes = 7, scale = False):
    # read in trials and label them
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    # force into proper arrays
    trainx = np.moveaxis(np.dstack(trials_padded), 2, 0)
    trainy = labs
    return trainx, trainy

# X, y = read_data("../PreTrainingDataset")

# also update this to have labels
# finally, we probably want to use keras generators, because its faster and
# memory efficient, check out
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
