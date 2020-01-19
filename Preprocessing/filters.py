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

from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt
from sklearn.preprocessing import MinMaxScaler

# Number references from sEMG feature extraction article
fs = 200.0
lowcut = 600.0
highcut = 20.0
# src: http://dx.doi.org/10.3390/s18051615
# Notch filtering is discouraged here 
# http://www.noraxon.com/wp-content/uploads/2014/12/ABC-EMG-ISBN.pdf

def butter_bandpass(lowcut=600, highcut=20, fs=2000, order=5):
    nyq = fs/2
    low = lowcut / nyq
    high = highcut / nyq
    b , a = butter(order, [high, low], btype='bandpass')
    return b,a

def butter_highpass(highcut=20, fs=200, order=4):
    nyq = fs/2
    high = highcut / nyq
    b , a = butter(order, high, btype='highpass')
    return b,a


def butter_bandpass_filter(data, lowcut=250, highcut=10, fs=2000, order=5):
    b,a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y

def butter_highpass_filter(data, highcut=20, fs=200, order=4):
    b,a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y