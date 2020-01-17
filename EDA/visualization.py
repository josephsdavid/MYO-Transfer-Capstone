#%%
from scipy.stats import gaussian_kde
import numpy as np
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import itertools as it
import pywt
from scipy.ndimage import zoom
import json
import os
import argparse


#%%

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5



# %%
def read_data(path, type):
    print("Reading Data")
    list_dataset = []
    list_labels = []


    for candidate in range(1):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Male' + str(candidate) + '/' + type + '/classe_%d.dat' % i,
                                              dtype=np.int8)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
            dataset_example = data_read_from_file
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        # examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    for candidate in range(1):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) + '/' + type + '/classe_%d.dat' % i,
                                              dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
            dataset_example = data_read_from_file
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        # examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    print("Finished Reading Data")
    return list_dataset, list_labels

# %%
def shape_series(arr, channels=8):
    _dim = (channels, int(len(arr)/channels))
    return np.reshape(arr, _dim)

def plot_trial(examples, labels, subject=0, classe=1, figsize=(5,8), filtered=False):
    
    groups = [0, 1, 2, 3, 5, 6, 7]
    session = shape_series(examples[subject][classe])
    gesture = shape_series(labels[subject][classe])[0][0]
    session2 = []
    if filtered==True:
        session2 = butter_bg_session(session)
        print(session.shape)

    fig, axes = pyplot.subplots(len(groups), 1,figsize=figsize, dpi=120)
    for n,group in enumerate(groups):
        axes[n].plot(session[group, :])
        if filtered:
            axes[n].plot(session2[group, :])
        axes[n].set_ylim(-150,150)
        axes[n].set_title('Channel {}'.format(group+1),y=0.8, loc='right')
    axes[n].set_xlabel('Gesture {} Code'.format(gesture))
    fig.show()







# %%
from scipy.signal import butter, lfilter, freqz
from sklearn.preprocessing import MinMaxScaler

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 1000.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bg_session(session):
    # Number references from sEMG feature extraction article
    # src: http://dx.doi.org/10.3390/s18051615
    # Notch filtering is discouraged here 
    # http://www.noraxon.com/wp-content/uploads/2014/12/ABC-EMG-ISBN.pdf
    fs = 200.0
    lowcut = 20.0
    highcut = 500.0

    # Time is sample count times freq of armband (200Hz)
    T = session.shape[1] * 1/fs
    nsamples = session.shape[1]
    t = np.linspace(0, T, nsamples, endpoint=False)

    session_b = [butter_bandpass_filter(x, lowcut, highcut, fs, order=4) for x in session]
    return np.array(session_b)

# %%
examples, labels = read_data('../PreTrainingDataset',type='training0')
plot_trial(examples, labels,0,20,(10,15))
plot_trial(examples, labels,0,20, (10,15), True)

# %%
