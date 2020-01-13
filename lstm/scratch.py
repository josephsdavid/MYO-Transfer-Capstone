from keras.models import load_model
from dataloaders import test_loader, val_loader
from scipy.stats import gaussian_kde
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import  read_data_augmented
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import itertools as it


class simple_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = LSTM(20, activation = act, name = 'LSTM_1', stateful = stateful, dropout = dropout, recurrent_dropout = dropout)(start)
        out = Dense(n_outputs, activation = 'softmax')(x)
        self.model = Model(start, out)

    def fit(self, X, y, lr = .005,
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
        adam = optimizers.adam(lr = lr)
        self.model.compile(optimizer = adam, **compilation_options)
        self.history = self.model.fit(X, y, **fit_options)


fit_options = {
    'epochs': 50,
    'batch_size':400,
    'shuffle':True,
    'verbose':1
}

# thankful for grid search homework
pars = ["scale","noise","filter", "shift"]
tunables = {key: [True, False] for key in pars}
grid = list(it.product(*(tunables[i] for i in tunables)))

results = {}
for i in range(len(grid)):
    hp = dict(zip(pars, grid[i]))
    X_test, y_test = test_loader("../EvaluationDataset", scale = grid[i][0])
    y_test = to_categorical(y_test)
    X_val, y_val = val_loader("../EvaluationDataset", scale = grid[i][0])
    y_val = to_categorical(y_val)
    X, y = read_data_augmented("../PreTrainingDataset", **hp)
    y = to_categorical(y)
    lstm = simple_lstm_classifier(X,y, dropout = 0.1)
    lstm.fit(X, y, lr = 0.0005, fit_options=fit_options)
    score = lstm.model.evaluate(X_val, y_val)[1]
    score0=lstm.model.evaluate(X_test, y_test)[1]
    results[grid[i]] = {"val":score,"test":score0}

results

