import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import utils
import load_pretrain as lp
from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt
#import tensorflow as tf



def butter_bandpass(lowcut=600, highcut=20, fs=2000, order=5):
    nyq = fs/2
    low = lowcut / nyq
    high = highcut / nyq
    b , a = butter(order, [high, low], btype='bandpass')
    return b,a


def butter_bandpass_filter(data, lowcut=600, highcut=20, fs=200, order=5):
    b,a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y

def butter_highpass(highcut=20, fs=200, order=4):
    nyq = fs/2
    high = highcut / nyq
    b , a = butter(order, high, btype='highpass')
    return b,a


def butter_highpass_filter(data, highcut=15, fs=200, order=4):
    b,a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y

def read_file(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    out = f.reshape(int(f.shape[0]/8),8)
    return out.astype(np.float32)


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


def reader_generator(instance):
    def read_group_to_lists(path, n_classes = 7):
        # grab in all the male candidates
        res = []
        labels = []
        trials = range(n_classes*4)
        for candidate in range(15):
            man = [read_file(path + '/Male' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
            # list addition is my new favorite python thing
            labs = [t % n_classes for t in trials]
            res += man
            labels += labs

        # and all the female candidates
        for candidate in range(2):
            woman = [read_file(path + '/Female' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
            labs = [t % n_classes for t in trials]
            res += woman
            labels += labs

        return res, labels
    return read_group_to_lists

train_reader, val_reader, test_reader = (reader_generator(x) for x in ['training0', 'Test0', 'Test1'])


def split(band=False):
    pt, labs = read_group_to_lists("../PreTrainingDataset")
    tr0, l0 = train_reader("../EvaluationDataset")
    tr1, l1 = val_reader("../EvaluationDataset")
    tr2, l2 = test_reader("../EvaluationDataset")
    res_emg = pt + tr0 + tr1 + tr2
    # fix to carson thing
    res_lab = labs + l0 + l1 + l2

    res_emg = [lp.butter_highpass_filter(x, 2, 200) for x in res_emg]
    res_emg = [lp.pad_along_axis(x, 1000, axis=0) for x in res_emg]

    split_emg, split_lab = ([x[:-56], x[-56:]] for x in [res_emg, res_lab])
    split_emg[0], split_lab[0]= lp.augment_data(split_emg[0], split_lab[0])
    split_emg[0] = split_emg[0] + [lp.add_noise_snr(i) for i in split_emg[0]]
    split_lab[0] = 2*split_lab[0]
    trainx = [lp.window_stack(x, 5, int(260/5)) for x in split_emg[0]]
    print(trainx[0].shape)
    testx = [lp.window_stack(x, 5, int(260/5)) for x in split_emg[1]]
    # force into proper arrays
    trainy = lp.roll_labels(trainx, split_lab[0])
    testy = lp.roll_labels(trainx, split_lab[1])
    train_labs = []
    # something something format for the stupid generator or dataset class
    #for i in range(len(split_lab[0])):
    #    train_labs.append(np.repeat(split_lab[0][i], split_emg[0][i].shape[0]))
    #test_labs = []
    #for i in range(len(split_lab[1])):
    #    test_labs.append(np.repeat(split_lab[1][i], split_emg[1][i].shape[0]))
    return np.moveaxis(np.concatenate(trainx,axis=0),2,1), np.hstack(trainy), np.moveaxis(np.concatenate(testx),2,1), np.hstack(testy)

xtr, ytr, xt, yt = split()





print(xtr.shape)

print(ytr.shape)

np.save('../data/x_train', xtr)
np.save('../data/x_val', xt)
np.save('../data/y_train', ytr)
np.save('../data/y_val', yt)




