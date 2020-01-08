import numpy as np
from load_pretrain import read_data, read_data_unrolled
from keras.layers import Dense, Dropout, LSTM, Input
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
import matplotlib.pyplot as plt

X, y = read_data("../PreTrainingDataset")
y = to_categorical(y)

# mn, mx = X.min(), X.max()
# X = (X - mn) / (mx - mn)

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})


# build the simplest lstm possible first
# then maybe stateful who knows
# try out embedding layers too




class simple_lstm_classifier:
    def __init__(self, X, y, act = 'relu', dropout = False):
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]
        start = Input((n_timesteps, n_features), name = 'Input')
        x = LSTM(100, activation = act, name = 'LSTM_1')(start)
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
                'epochs': 20,
                'batch_size':300,
                'verbose':1
            }):
        self.model.compile(**compilation_options)
        self.history = self.model.fit(X, y, **fit_options)


lstm = simple_lstm_classifier(X, y)
lstm.fit(X,y)

history = lstm.history


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# with generator
