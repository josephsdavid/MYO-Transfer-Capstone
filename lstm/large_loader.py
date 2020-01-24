import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import utils
import load_pretrain as lp
from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt
import tensorflow as tf

tf.data.Dataset.from_tensors(x)


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
    res_emg = [lp.butter_highpass_filter(x, 2, 200) for x in res_emg]
    res_lab = labs + l0 + l1 + l2
    split_emg, split_lab = ([x[:-28], x[-28:]] for x in [res_emg, res_lab])
    train_labs = []
    # something something format for the stupid generator or dataset class
    for k in range(len(res_lab[0])):
        train_labs[i] = np.repeat:w

    return split_emg, split_lab

x, y = split()

len(x)

len(x[0])

len(x[1])


