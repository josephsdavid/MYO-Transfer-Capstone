import numpy as np
from load_pretrain import  read_data_filtered_augmented
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional
from keras.layers import Embedding, Activation, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import TensorBoard
import math
import matplotlib.pyplot as plt



X, y = read_data_filtered_augmented("../PreTrainingDataset")
y = to_categorical(y)

X.shape


# I dont think there is much benefit to stacking LSTM layers in our case. The
# idea of a stacked RNN is to evaluate the data at different timescales, and I
# believe that this may not be beneficial in our approach

class stacked_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = (LSTM(20,
                               activation = act,
                               name = 'LSTM_1',
                               stateful = stateful,
                               dropout = dropout,
                               recurrent_dropout = dropout,
                               return_sequences=True))(start)
        x = (LSTM(40,
                               activation = act,
                               name = 'LSTM_2',
                               stateful = stateful,
                               dropout = dropout,
                               recurrent_dropout = dropout))(x)
        out = Dense(n_outputs, activation = 'softmax')(x)
        self.model = Model(start, out)

    def fit(self, X, y, lr = .001,
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'metrics' : ['accuracy']},
            fit_options = {
                'epochs': 50,
                'batch_size':400,
                'validation_split':0.33,
                'shuffle':False,
                'verbose':1
            }, tensorboard = True):
        if tensorboard:
            callbacks = TensorBoard(batch_size = fit_options['batch_size'], update_freq=4000 , histogram_freq=1)
            adam = optimizers.adam(lr = lr)
            self.model.compile(optimizer = adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks])
        else:
            adam = optimizers.adam(lr = lr)
            self.model.compile(optimizer = adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options)

lstm = stacked_lstm_classifier(X, y, dropout = 0.1)

lstm.fit(X,y, tensorboard = False)

history = lstm.history

plt.subplot(212)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def calculate_nodes(X, y, alpha = 2):
    Ns = X.shape[0]
    Ni = X.shape[2]
    No = y.shape[-1]
    Nh = Ns / (alpha * (Ni + No))
    return math.floor(Nh)

# I think this should

class wide_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, batch_norm = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = Bidirectional(LSTM(60, activation = act,name = 'LSTM_1',
                  dropout = dropout,
                  recurrent_dropout = dropout))(start)
        if batch_norm:
            x = BatchNormalization()(x)
        out = Dense(n_outputs, activation = 'softmax')(x)
        self.model = Model(start, out)

    def fit(self, X, y, lr = .001,
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'metrics' : ['accuracy']},
            fit_options = {
                'epochs': 50,
                'batch_size':400,
                'validation_split':0.33,
                'shuffle':False,
                'verbose':1
            }, tensorboard = True):
        if tensorboard:
            callbacks = TensorBoard(batch_size = fit_options['batch_size'], update_freq=4000 , histogram_freq=1)
            adam = optimizers.adam(lr = lr)
            self.model.compile(optimizer = adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks])
        else:
            adam = optimizers.adam(lr = lr)
            self.model.compile(optimizer = adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options)


lstm = wide_lstm_classifier(X, y, dropout = 0.1, batch_norm = False)

lstm.fit(X,y, tensorboard=False, lr = 0.0005)

history = lstm.history

print(history.history['val_acc'])


plt.subplot(212)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2)
plt.savefig("wide_lstm_training.png")
plt.show()
