#%%
from keras.models import load_model
from scipy.stats import gaussian_kde
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.applications import Xception
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import itertools as it

import json
import os
import argparse


#%%
import pywt
from scipy.ndimage import zoom
number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5

def calculate_wavelet_dataset(dataset):
    dataset_spectrogram = []
    mother_wavelet = 'mexh'
    for examples in dataset:
        canals = []
        for electrode_vector in examples:
            coefs = calculate_wavelet_vector(np.abs(electrode_vector), 
                                            mother_wavelet=mother_wavelet, 
                                            scales=np.arange(1, 33))  # 33 originally
            # print(np.shape(coefs))
            # show_wavelet(coef=coefs)
            coefs = zoom(coefs, .25, order=0)
            coefs = np.delete(coefs, axis=0, obj=len(coefs)-1)
            coefs = np.delete(coefs, axis=1, obj=np.shape(coefs)[1]-1)
            canals.append(np.swapaxes(coefs, 0, 1))
        example_to_classify = np.swapaxes(canals, 0, 1)
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram

def calculate_wavelet_vector(vector, mother_wavelet='mexh', scales=np.arange(1, 32)):
    coef, freqs = pywt.cwt(vector, scales=scales, wavelet=mother_wavelet)
    return coef

def show_wavelet(coef):
    print(np.shape(coef))
    plt.rcParams.update({'font.size': 36})
    plt.matshow(coef)
    plt.ylabel('Scale')
    plt.xlabel('Samples')
    plt.show()



#%%
def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (example == []):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example):
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    data_calculated = calculate_wavelet_dataset(dataset_example_formatted)
    return np.array(dataset_example_formatted)

# %%
def read_data(path, type):
    print("Reading Data")
    list_dataset = []
    list_labels = []


    for candidate in range(15):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Male' + str(candidate) + '/' + type + '/classe_%d.dat' % i,
                                              dtype=np.int8)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        # examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    for candidate in range(2):
        labels = []
        examples = []
        for i in range(number_of_classes * 4):
            data_read_from_file = np.fromfile(path + '/Female' + str(candidate) + '/' + type + '/classe_%d.dat' % i,
                                              dtype=np.int8)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
            dataset_example = format_data_to_train(data_read_from_file)
            examples.append(dataset_example)
            labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
        # examples, labels = shift_electrodes(examples, labels)
        list_dataset.append(examples)
        list_labels.append(labels)

    print("Finished Reading Data")
    return list_dataset, list_labels
# %%
examples, labels = read_data('../EvaluationDataset',type='training0')
labels.shape()
# %%
from matplotlib import pyplot
# load dataset
values = examples
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	i += 1
pyplot.show()

# %%
