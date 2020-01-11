import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import load_pretrain as lp


# loaders for non NINAPro data
# fix for the number of men and women in each dataset

# the output of this function is a list of (timesteps, features) arrays, as well
# as some labels
def reader_generator(instance):
    def read_group_to_lists(path, n_classes = 7):
        # grab in all the male candidates
        res = []
        labels = []
        trials = range(n_classes*4)
        for candidate in range(15):
            man = [lp.read_file(path + '/Male' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
            # list addition is my new favorite python thing
            labs = [t % n_classes for t in trials]
            res += man
            labels += labs

        # and all the female candidates
        for candidate in range(2):
            woman = [lp.read_file(path + '/Female' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
            labs = [t % n_classes for t in trials]
            res += woman
            labels += labs

        return res, labels
    return read_group_to_lists

train_reader = reader_generator('training0')
val_reader = reader_generator('Test0')
test_reader = reader_generator('Test1')


def train_loader(path, n_classes = 7, scale = False):
    # read data [(time, feat)]
    trials_all, labs  = train_reader(path, n_classes = n_classes)
    # shift electrodes for double data
    trials_all, labs = lp.augment_data(trials_all, labs)
    # add noisy data while preserving SNR of 25, doubling data again
    trials_all = trials_all + [lp.add_noise_snr(i) for i in trials_all]
    labs = labs + labs
    # butterworth
    trials_all = [lp.butter_highpass_filter(x, 2, 200) for x in trials_all]
    if (scale):
        trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
    # get maximum length for padding
    maxlen = max([x.shape[0] for x in trials_all])
    # pad data for the lstm
    trials_padded = [lp.pad_along_axis(x, maxlen, axis=0) for x in trials_all]
    # sliding window trials
    trials_rolled = [lp.window_stack(x, 21, int(260/5)) for x in trials_padded]
    # force into proper arrays
    trainy = lp.roll_labels(trials_rolled, labs)
    trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
    return trainx, trainy

def eval_generator(fn):
    def read_data_filtered(path, n_classes = 7, scale = False):
        # read data [(time, feat)]
        trials_all, labs  = fn(path, n_classes = n_classes)
        trials_all = [lp.butter_highpass_filter(x, 2, 200) for x in trials_all]
        if (scale):
            trials_all = [MinMaxScaler().fit_transform(x) for x in trials_all]
        # get maximum length for padding
        maxlen = max([x.shape[0] for x in trials_all])
        # pad data for the lstm
        trials_padded = [lp.pad_along_axis(x, maxlen, axis=0) for x in trials_all]
        # sliding window trials
        trials_rolled = [lp.window_stack(x, 21, int(260/5)) for x in trials_padded]
        # force into proper arrays
        trainy = lp.roll_labels(trials_rolled, labs)
        trainx = np.moveaxis(np.concatenate(trials_rolled, axis = 2), 2, 0)
        return trainx, trainy
    return(read_data_filtered)


val_loader, test_loader = (eval_generator(f) for f in [val_reader, test_reader])

