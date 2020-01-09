import numpy as np
from scipy import signal
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
tr, _ = read_group_to_lists("../PreTrainingDataset")


def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.dstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )




def pad_along_axis(array: np.ndarray, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)
    if pad_size < 0:
        return array
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b




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
    trials_rolled = [window_stack(x, 5, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

# for use with TimeSeries Generator, this is crap
def read_data_unrolled(path, n_classes = 7):
    # read in trials and label them
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    # force into proper arrays
    trainx = np.moveaxis(np.dstack(trials_padded), 2, 0)
    trainy = labs
    return trainx, trainy


def butter_highpass(cutoff, fs, order=3):
    # nyquist frequency!!
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# trials_all, _ = read_group_to_lists("../PreTrainingDataset")
#
# x = trials_all[0]
#
# import matplotlib.pyplot as plt
#
# plt.plot(x)
# plt.savefig("unfilt")
# plt.plot(butter_highpass_filter(x, 2, 200))
# plt.savefig("filt")

def read_data_filtered(path, n_classes = 7, scale = False):
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    trials_all = [butter_highpass_filter(x, 2, 200) for x in trials_all]
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])

    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    trials_rolled = [window_stack(x, 5, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

# X, y = read_data("../PreTrainingDataset")

# also update this to have labels
# finally, we probably want to use keras generators, because its faster and
# memory efficient, check out
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
