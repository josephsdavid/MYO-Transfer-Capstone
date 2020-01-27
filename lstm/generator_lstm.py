from tensorflow.keras.models import load_model
from dataloaders import test_loader
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
import matplotlib.pyplot as plt

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

x_tr = np.load("../data/x_train.npy")
y_tr = np.load("../data/y_train.npy")
x_val = np.load("../data/x_val.npy")
y_val = np.load("../data/y_val.npy")

y_tr = to_categorical(y_tr)
y_val = to_categorical(y_val)


train_gen = TimeseriesGenerator(x_tr, y_tr, length = int(260/5), stride = 5 , shuffle = True, batch_size = 500)
val_gen = TimeseriesGenerator(x_val, y_val, length = int(260/5), stride = 5 , shuffle = True, batch_size = 500)


class simple_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False):
        start = Input((None, X.shape[1]), name = 'Input')
        x = LSTM(20, activation = act, name = 'LSTM_1', stateful = stateful, dropout = dropout, recurrent_dropout = dropout)(start)
        out = Dense(y.shape[1], activation = 'softmax')(x)
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
        self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks])


lstm = simple_lstm_classifier(x_tr, y_tr, dropout = 0.5)


compilation_options = {
    'loss' : 'categorical_crossentropy',
    'metrics' : ['accuracy']}
Adam = optimizers.Adam(lr = 0.0005)
lstm.model.compile(optimizer=Adam, **compilation_options)



lstm.model.fit(train_gen, validation_data = val_gen, epochs = 100)

lstm.fit(train_gen, lr = 0.0005, fit_options=fit_options,  epochs = 100)


lstm.model.save("models/simple_lstm.h5")
