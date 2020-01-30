import numpy as np


def roll_labels(x, y):
    labs_rolled = []
    for i in range(len(y)):
        l = y[i]
        n = x[i].shape[0]
        labs_rolled.append(np.repeat(l,n))
    return np.hstack(labs_rolled)

def window_roll(a, stepsize=5, width=52):
    n = a.shape[0]
    emg =  np.dstack( [a[i:1+n+i-width:stepsize] for i in range(0,width)] )
    return emg


"""
augmentors should take in a list of arrays and ouptut a list of bigger arrays,
arguments should be (X, augmentors should take in a list of arrays and ouptut a list of bigger arrays,
arguments should be (X, y)

"""


def _add_noise_snr(signal, snr = 25):
    # convert signal to db
    sgn_db = np.log10((signal ** 2).mean(axis = 0))  * 10
    # noise in db
    noise_avg_db = sgn_db - snr
    # convert noise_db
    noise_variance = 10 ** (noise_avg_db /10)
    # make some white noise using this as std
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)
    return(signal + noise)

def add_noise(x, y, snr=25):
    x2 = []
    for i in range(len(x)):
        x2.append(_add_noise_snr(x[i]))
    x = x + x2
    y = y*2
    return x, y


