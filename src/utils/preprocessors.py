from scipy import signal
from sklearn.preprocessing import StandardScaler
import pywt
from PyEMD import EMD

def _butter_highpass(cutoff, fs, order=3):
    # nyquist frequency!!
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff=2, fs=200, order=3):
    b, a = _butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def scale(X):
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = StandardScaler().fit_transform(X[i, :, :])

    return X

