import numpy as np
from load_pretrain import read_data, read_data_unrolled
from keras.layers import Dense, Dropout, LSTM, Input
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

X, y = read_data("../PreTrainingDataset")
y = to_categorical(y)



# build the simplest lstm possible first
# then maybe stateful who knows
# try out embedding layers too
class simple_lstm_classifier:
    def __init__(self, X, y, act = 'relu', dropout = False, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = LSTM(8, activation = act, name = 'LSTM_1', stateful = stateful)(start)
        if (dropout):
            x = Dropout(0.5)(x)
        x = Dense(n_outputs, activation = 'relu')(x)
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


lstm = simple_lstm_classifier(X, y, dropout = True)
lstm.fit(X,y)


history = lstm.history


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# this is a work in progress clearly
class bigger_lstm_classifier:
    def __init__(self, X, y, act = 'relu', dropout = False, n_layers = 2):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = LSTM(8, activation = act, name = 'LSTM_1', stateful = stateful, return_sequences = True)(start)
        for
        if (dropout):
            x = Dropout(0.5)(x)
        x = LSTM(16, activation = act, name = 'LSTM_2')
        x = Dense(n_outputs, activation = 'relu')(x)
        self.model = Model(start, x)

    def fit(self, X, y,
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'optimizer' : 'adam',
                'metrics' : ['accuracy']},
            fit_options = {
                'epochs': 20,
                'batch_size':300,
                'verbose':1
            }):
        self.model.compile(**compilation_options)
        self.history = self.model.fit(X, y, **fit_options)
