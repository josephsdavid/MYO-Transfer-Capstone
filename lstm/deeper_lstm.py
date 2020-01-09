import numpy as np
from load_pretrain import read_data, read_data_unrolled, read_data_filtered
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional
from keras.layers import Embedding, Activation, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt



X, y = read_data_filtered("../PreTrainingDataset")
y = to_categorical(y)

X.shape

# this is a pretty decent classifier, I saved the plot so we can see how it
# works.

class lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((None, n_features), name = 'Input')
        x = Bidirectional(LSTM(20, activation = act, name = 'LSTM_1', stateful = stateful, dropout = dropout, recurrent_dropout = dropout))(start)
        x = Dense(40, activation =act)(x)
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
        adam = optimizers.adam(lr = lr)
        self.model.compile(optimizer = adam, **compilation_options)
        self.history = self.model.fit(X, y, **fit_options)





lstm = lstm_classifier(X, y, dropout = 0.5)
lstm.fit(X,y)

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


