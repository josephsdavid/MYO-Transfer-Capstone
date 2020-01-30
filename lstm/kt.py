import kerastuner as kt
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataloaders import test_loader
from scipy.stats import gaussian_kde
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import read_data, read_data_augmented, read_data_filtered
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.layers import Embedding, Activation, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.regularizers as rr
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt



x_tr = np.load("../data/x_train.npy")
y_tr = np.load("../data/y_train.npy")
x_val = np.load("../data/x_val.npy")
y_val = np.load("../data/y_val.npy")


def build_model(hp):
    n_lstms = hp.Int('lstm_layers', 1, 3, default = 1)
    ns = range(n_lstms)
    inputs = Input((52, 8))
    x = inputs
    for i in ns:
        seq = False if (i == (len(ns) -1)) else True
        x = LSTM(
            units=hp.Int('units_' + str(i), 20, 200, step = 30),
            dropout = 0.5,
            recurrent_dropout = 0.5,
            return_sequences = seq, activation = 'tanh'
        )(x)

    outputs = Dense(7, activation = 'softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                            values = [1e-2, 1e-3, 1e-4])),
                                  loss = 'sparse_categorical_crossentropy',
                                  metrics = ['accuracy']
    )
    return model


tuner = kt.BayesianOptimization(build_model, objective = 'val_accuracy', max_trials = 25, executions_per_trial = 2, directory = 'result', project_name = 'lstm')


tuner.search(x_tr, y_tr, epochs = 5, validation_data = (x_val, y_val), batch_size = 3000)


