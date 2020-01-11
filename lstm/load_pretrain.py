import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

# load up the first example first!

def read_file(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    return f.reshape(int(f.shape[0]/8),8)


# the output of this function is a list of (timesteps, features) arrays, as well
# as some labels
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


# this is the key sliding window function, stepsize is the distance apart each
# window is (so for example with the defaults, if the data is measure in
# seconds, we are going to take 3 second windows with 2 (width-stepsize) overlap)
# This function takes in an array of (timesteps, features), and outputs an array
# of (time related number, features, samples). This has the bonus of
# preformatting the data for an LSTM
def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.dstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )



# pads along the axis of a numpy array, to make them all the same length, that
# way our list can be dstacked into an array
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
# trials_rolled = [window_stack(x, 21, int(260/5)) for x in trials_padded]

# to get the labels to match up with our windows
def roll_labels(x, y):
    labs_rolled = []
    for i in range(len(y)):
        l = y[i]
        n = x[i].shape[2]
        labs_rolled.append(np.repeat(l,n))
    return np.hstack(labs_rolled)


def read_data(path, n_classes = 7, scale = False):
    # read in trials and label them, data format is list of (timesteps,
    # features) arrays
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    # new data format is a list of (time related deal, features, samples) arrays
    trials_rolled = [window_stack(x, 21, int(260/5)) for x in trials_padded]
    trainy = roll_labels(trials_rolled, labs)
    # then we just concatenate the list, formatting the data as a new numpy
    # array with dims (samples, time, features)
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


# high pass butterworth filter discussed in the paper
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
    # same thing as read_data but its got a filter
    trials_all = [butter_highpass_filter(x, 2, 200) for x in trials_all]
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])

    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    trials_rolled = [window_stack(x, 21, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

# X, y = read_data("../PreTrainingDataset")

# also update this to have labels
# finally, we probably want to use keras generators, because its faster and
# memory efficient, check out
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly



# electrode shifting data aug
def shift_electrodes(examples, labels):
    index_normal_class = [1, 2, 6, 2]  # The normal activation of the electrodes.
    class_mean = []
    # For the classes that are relatively invariant to the highest canals activation, we get on average for a
    # subject the most active canals for those classes
    for classe in range(3, 7):
        X_example = []
        Y_example = []
        for k in range(len(examples)):
            X_example.append(examples[k])
            Y_example.append(labels[k])

        cwt_add = []
        for j in range(len(X_example)):
            if Y_example[j] == classe:
                if cwt_add == []:
                    cwt_add = np.array(X_example[j][0])
                else:
                    cwt_add += np.array(X_example[j][0])
        class_mean.append(np.argmax(np.sum(np.array(cwt_add), axis=0)))

    # We check how many we have to shift for each channels to get back to the normal activation
    new_cwt_emplacement_left = ((np.array(class_mean) - np.array(index_normal_class)) % 10)
    new_cwt_emplacement_right = ((np.array(index_normal_class) - np.array(class_mean)) % 10)

    shifts_array = []
    for valueA, valueB in zip(new_cwt_emplacement_left, new_cwt_emplacement_right):
        if valueA < valueB:
            # We want to shift toward the left (the start of the array)
            orientation = -1
            shifts_array.append(orientation*valueA)
        else:
            # We want to shift toward the right (the end of the array)
            orientation = 1
            shifts_array.append(orientation*valueB)

    # We get the mean amount of shift and round it up to get a discrete number representing how much we have to shift
    # if we consider all the canals
    # Do the shifting only if the absolute mean is greater or equal to 0.5
    final_shifting = np.mean(np.array(shifts_array))
    if abs(final_shifting) >= 0.5:
        final_shifting = int(np.round(final_shifting))
    else:
        final_shifting = 0

    # Build the dataset of the candiate with the circular shift taken into account.
    X_example = []
    Y_example = []
    for k in range(len(examples)):
        sub_ensemble_example = []
        for example in examples[k]:
            sub_ensemble_example.append(np.roll(np.array(example), final_shifting))
        X_example.append(sub_ensemble_example)
        Y_example.append(labels[k])
    return X_example, Y_example

def augment_data(trials, labs):
    x, y = shift_electrodes(trials, labs)
    labs.append(y)
    for i in range(len(x)):
        trials.append(np.vstack(x[i]))
    return trials, labs



def add_noise_snr(signal, snr = 25):
    # convert signal to db
    sgn_db = np.log10((signal ** 2).mean(axis = 0))  * 10
    # noise in db
    noise_avg_db = sgn_db - snr
    # convert noise_db
    noise_variance = 10 ** (noise_avg_db /10)
    # make some white noise using this as std
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)
    return(signal + noise)

# x, y = read_group_to_lists("../PreTrainingDataset")
#
# x2 = [add_noise_snr(i) for i in x]
#
# [i.shape for i in x2]

def read_data_filtered_augmented(path, n_classes = 7, scale = False):
    # read data [(time, feat)]
    trials_all, labs  = read_group_to_lists(path, n_classes = n_classes)
    # shift electrodes for double data
    trials_all, labs = augment_data(trials_all, labs)
    # add noisy data while preserving SNR of 25, doubling data again
    trials_all = trials_all + [add_noise_snr(i) for i in trials_all]
    labs = labs + labs
    # butterworth
    trials_all = [butter_highpass_filter(x, 2, 200) for x in trials_all]
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    # pad data for the lstm
    trials_padded = [pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    trials_rolled = [window_stack(x, 21, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

