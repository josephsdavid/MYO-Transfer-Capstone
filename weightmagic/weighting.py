from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import read_data, read_data_augmented, read_data_filtered
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.layers import Embedding, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())


X, y = read_data_augmented("../PreTrainingDataset")
y = to_categorical(y)


class simple_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = LSTM(20, activation = act, name = 'LSTM_1', stateful = stateful, dropout = dropout, recurrent_dropout = dropout)(start)
        out = Dense(n_outputs, activation = 'softmax')(x)
        self.model = Model(start, out)

    def fit(self, X, y, lr = .001,
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'metrics' : ['accuracy']},
            fit_options = {
                'epochs': 100,
                'batch_size':400,
                'validation_split':0.33,
                'shuffle':False,
                'verbose':1
            }):
        Adam = optimizers.Adam(lr = lr)
        self.model.compile(optimizer = Adam, **compilation_options)
        callbacks = EarlyStopping(monitor = 'accuracy', patience = 10, mode = 'max')
        self.cbk = CollectWeightCallback(layer_index=-1)
        self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks, self.cbk])


lstm = simple_lstm_classifier(X, y, dropout = 0.1)

fit_options = {
    'epochs': 100,
    'batch_size':400,
    'shuffle':True,
    'verbose':2
}

lstm.fit(X, y, lr = 0.0005, fit_options=fit_options)

lstm_weights = lstm.cbk.weights[-1][:-2]

lstm_weights.shape
def glorot_norm_fix(self, W, N, M, rf_size):
    """Apply Glorot Normalization Fix"""

        kappa = np.sqrt( 2 / ((N + M)*rf_size) )
        W = W/kappa
        return W


def glorot_norm_check(self, W, N, M, rf_size,
                      lower = 0.5, upper = 1.5):
    """Check if this layer needs Glorot Normalization Fix"""
        kappa = np.sqrt( 2 / ((N + M)*rf_size) )
        norm = np.linalg.norm(W)
        check1 = norm / np.sqrt(N*M)
        check2 = norm / (kappa*np.sqrt(N*M))
        if (rf_size > 1) and (check2 > lower) and (check2 < upper):
            return check2, True
        elif (check1 > lower) & (check1 < upper):
            return check1, True
        else:
            if rf_size > 1:
                return check2, False
            else:
                return check1, False


def analyze_weights(weights, min_size, max_size, alphas, lognorms, spectralnorms,
                    softranks, normalize, glorot_fix, plot, mp_fit):
    from sklearn.decomposition import TruncatedSVD
    res = {}
    count = len(weights)
    if count == 0:
        return res
    for i, W in enumerate(weights):
        res[i] = {}
        M, N = np.min(W.shape), np.max(W.shape)
        Q = N/M
        res[i]["N"] = N
        res[i]["M"] = M
        res[i]["Q"] = Q
        l0 = None
        check, checkTF = self.glorot_norm_check(W, N, M, count)
        res[i]['check'] = check
        res[i]['checkTF'] = checkTF
        # assume receptive field size is count
        if glorot_fix:
            W = self.glorot_norm_fix(W, N, M, count)
        else:
            # probably never needed since we always fix for glorot
            W = W * np.sqrt(count/2.0)
        if spectralnorms: #spectralnorm is the max eigenvalues
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=10)
            svd.fit(W)
            sv = svd.singular_values_
            sv_max = np.max(sv)
            evals = sv*sv
            if normalize:
                evals = evals/N
            l0 = evals[0]
            res[i]["spectralnorm"] = l0
            res[i]["logspectralnorm"] = np.log10(l0)
