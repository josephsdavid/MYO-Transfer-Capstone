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
import re
from pathlib import Path

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5

def shape_series(arr, channels=8):
    _dim = (channels, int(len(arr)/channels))
    return np.reshape(arr, _dim)

def read_data(path, type):
    print("Reading Data")
    list_dataset = []
    list_labels = []

    basepath = Path(os.getcwd()+'/'+path)
    pattern_m = '^Male(\d)'
    pattern_f = '^Female(\d)'
    nMax = 0
    nMin = 1e10

    xMax = []
    xMaxPerson = []
    for entry in basepath.iterdir():
        if re.match(pattern_m, entry.name) and entry.is_dir():
            labels = []
            examples = []
            for i in range(number_of_classes * 4):
                fpath = path + '/' + entry.name + '/' + type + '/classe_%d.dat' % i
                data_read_from_file = np.fromfile(fpath, dtype=np.int8)
                data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
                dataset_example = data_read_from_file - np.mean(data_read_from_file) # shape_series(data_read_from_file)
                np.clip(dataset_example, -128, 127, out=dataset_example)
                examples.append(dataset_example)
                
                labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            n_size = int(len(examples[0])/8)
            
            if(n_size>nMax):
                nMax = n_size
            if(n_size<nMin):
                nMin = n_size
            list_dataset.append(examples)
            list_labels.append(labels)
        elif re.match(pattern_f,entry.name) and entry.is_dir():
            labels = []
            examples = []
            for i in range(number_of_classes * 4):
                fpath = path + '/' + entry.name + '/' + type + '/classe_%d.dat' % i
                data_read_from_file = np.fromfile(fpath, dtype=np.int8)
                data_read_from_file = np.array(data_read_from_file, dtype=np.float16)
                dataset_example = data_read_from_file - np.mean(data_read_from_file) # shape_series(data_read_from_file)
                np.clip(dataset_example, -128, 127, out=dataset_example)
                examples.append(dataset_example)
                print(len(dataset_example), len(dataset_example)//8)
                labels.append((i % number_of_classes) + np.zeros(dataset_example.shape[0]))
            n_size = int(len(examples[0])/8)
            if(n_size>nMax):
                nMax = n_size
            if(n_size<nMin):
                nMin = n_size
            list_dataset.append(examples)
            list_labels.append(labels)
    # print('nMin: %d'%(nMin), 'nMax: %d'%(nMax))
    return np.array(list_dataset), np.array(list_labels)



# %%
# examples, labels =  read_data('PreTrainingDataset', type='training0')
# print(examples.shape)
# %%
