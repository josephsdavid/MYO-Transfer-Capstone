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
from Preprocessing.filters import butter_bandpass_filter, butter_highpass_filter
from Preprocessing.data_utils import shape_series
from Preprocessing.transforms import to_freq
from scipy import signal
#%%

number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5

fs = 200.0
lowcut = 450.0
highcut = 20.0
groups = [0, 1, 2, 3, 5, 6, 7]



def calculate_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    fss, tss, sov = signal.spectrogram(x=vector, fs=fs,
                    nperseg=npserseg,
                    noverlap=noverlap,
                    window="hann",
                    scaling="spectrum")
                                                                                        

    frequencies_samples, time_segment_sample, spectrogram_of_vector = fss, tss, sov
    return spectrogram_of_vector, time_segment_sample, frequencies_samples

def show_spectrogram(frequencies_samples, time_segment_sample, spectrogram_of_vector, ax):
    ax.pcolormesh(time_segment_sample, frequencies_samples, spectrogram_of_vector, cmap='viridis')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [ms]')
    # ax.set_title("STFT")
    return ax


# %%
def butter_bg_session(session):
    # Time is sample count times freq of armband (200Hz)
    session_b = [butter_highpass_filter(x,  order=4) for x in session]
    return np.array(session_b)

def plot_f(x, Y, labels, title='',  figsize=(5,8)):
    fig, axes = plt.subplots(len(labels), 1,figsize=figsize)
    xticks = np.arange(0,101, step=10)
    for n,l in enumerate(labels):
        for a in Y:
            N = a[l,:].size
            xf, fft= to_freq(a[l, :], fs)
            fft_mag = np.abs(fft)
            fft_db = 20 * np.log10(fft_mag)
            axes[n].plot(xf[:N // 2], fft_db[:N//2])
            axes[n].set_xticks(xticks)
            axes[n].set_ylabel('PSD (dB)')
        axes[n].set_title('Channel {}'.format(l+1),y=0.8, loc='right')
    axes[n].set_xlabel(title)
    
    fig.show()


def plot_ts(x, Y, labels, title='',  figsize=(5,8)):
    fig, axes = plt.subplots(len(labels), 1,figsize=figsize)
    yticks = np.arange(-130,131, step=20)
    for n,l in enumerate(labels):
        for a in Y:
            axes[n].plot(x,a[l, :])
            axes[n].set_yticks(yticks)
            axes[n].set_ylim(-130,130)
        axes[n].set_title('Channel {}'.format(l+1),y=0.8, loc='right')
    axes[n].set_xlabel(title)
    fig.show()

def plot_spectro(x, Y, labels, title='',  figsize=(5,8)):
    fig, axes = plt.subplots(len(labels), 1,figsize=figsize)
    for n,l in enumerate(labels):
        for a in Y:
            sv, tss, fqs = calculate_spectrogram_vector(a[l, :])
            # sv = np.swapaxes(sv,0,1)
            axes[n] = show_spectrogram(fqs, tss, sv, axes[n])
    fig.show()

plots_functions = {
    'freq': plot_f,
    'ts': plot_ts,
    'spec': plot_spectro
}

def plot_trial(examples, labels, subject=0, classe=1, figsize=(5,8), filtered=False, ptype='ts'):
    if type(classe)!= int:
        session = []
        for i in classe:
            temp = shape_series(examples[subject][i]).T
            session.append(temp[:1000,:])
        session = np.concatenate(session, ).T
        # print(session.shape)
        title = ''
    else:
        session = shape_series(examples[subject][classe])
        gesture = shape_series(labels[subject][classe])[0][0]
        title = 'Gesture {} Code'.format(gesture)
    
    
    nsamples = session.shape[1]
    T = nsamples * 1/fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    # print(nsamples, 1/fs)
    data = [session]
    if filtered==True:
        data.append(butter_bg_session(session))

    
    plots_functions[ptype](t, data, groups, title, figsize)
    





# %%
def calculate_spectrogram_dataset(dataset):
    dataset_spectrogram = []
    for examples in dataset:
        canals = []
        examples = shape_series(examples)
        for electrode_vector in examples:
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            #remove the low frequency signal as it's useless for sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))

        example_to_classify = np.swapaxes(canals, 0, 1)
        dataset_spectrogram.append(example_to_classify)

    return dataset_spectrogram


