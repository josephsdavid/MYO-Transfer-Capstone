import numpy as np
from load_pretrain import read_data, read_data_unrolled
from keras.layers import Dense, Dropout, LSTM, Input
from keras.layers import Embedding, Activation, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

X, y = read_data("../PreTrainingDataset")
y = to_categorical(y)

X.shape



# build the simplest lstm possible first
# then maybe stateful who knows
# try out embedding layers too
class lstm_classifier:
    def __init__(self, X, y, act = 'relu', dropout = True, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = LSTM(8, activation = act, name = 'LSTM_1', return_sequences = True, dropout = 0.5, recurrent_dropout = 0.5)(start)
        x = LSTM(16, activation = act, name = 'LSTM_2', dropout = 0.5, recurrent_dropout = 0.5)(x)
        for i in range(3):
            x = Dropout(0.5)(x)
            x = Dense(32, activation = act)(x)
        x = Dropout(0.5)(x)
        x = Dense(n_outputs, activation = 'relu', name = 'output')(x)
        self.model = Model(start, x)

    def fit(self, X, y,
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'optimizer' : 'adam',
                'metrics' : ['accuracy']},
            fit_options = {
                'epochs': 100,
                'batch_size':300,
                'validation_split':0.33,
                'shuffle':False,
                'verbose':1
            }):
        self.model.compile(**compilation_options)
        self.history = self.model.fit(X, y, **fit_options)


lstm = lstm_classifier(X, y)
lstm.fit(X,y)
