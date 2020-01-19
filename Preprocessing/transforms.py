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

from scipy import fftpack

def to_freq(x, f_s=200):
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x),d=1/2000)
    # 1/T = frequency
    f = np.linspace(0, f_s, len(x))
    return f, X
