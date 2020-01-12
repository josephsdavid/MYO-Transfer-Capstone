from keras.models import load_model
from dataloaders import test_loader
from scipy.stats import gaussian_kde
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import read_data, read_data_filtered_augmented, read_data_filtered
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

X, y = read_data_filtered_augmented("../PreTrainingDataset", noise = False)
# X, y = read_data("../PreTrainingDataset")
y = to_categorical(y)


# calculate number of nodes!
def calculate_nodes(X, y, alpha = 2):
    Ns = X.shape[0]
    Ni = X.shape[2]
    No = y.shape[-1]
    Nh = Ns / (alpha * (Ni + No))
    return(Nh)

calculate_nodes(X, y, 2)


# build the simplest lstm possible first
# then maybe stateful who knows
# try out embedding layers too




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
        adam = optimizers.adam(lr = lr)
        self.model.compile(optimizer = adam, **compilation_options)
        self.history = self.model.fit(X, y, **fit_options)


lstm = simple_lstm_classifier(X, y, dropout = 0.1)

fit_options = {
    'epochs': 100,
    'batch_size':400,
    'shuffle':True,
    'verbose':1
}

lstm.fit(X, y, lr = 0.0005, fit_options=fit_options)


lstm.model.save("models/simple_lstm.h5")


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
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2)
plt.savefig("simple_lstm_training.png")
plt.show()



######################## temporary evaluation
X_test, y_test = test_loader("../EvaluationDataset")
y_test = to_categorical(y_test)
X_test.shape

score = lstm.model.evaluate(X_test, y_test)
# 68% accuracy


preds = lstm.model.predict(X_test)


fig = plt.figure()
for k in range(preds.shape[-1]):
    ax = fig.add_subplot(3, 3, k+1)
    ax.plot(np.linspace(0,1, 200),gaussian_kde(preds[:,k])(np.linspace(0,1,200)), label = k)
    ax.set_title(str(k))
plt.savefig('simple_lstm_class_probs.png')
plt.show()


fig = plt.figure()
for k in range(preds.shape[-1]):
    ax = fig.add_subplot(3, 3, k+1)
    ax.plot(np.linspace(0,1, 200),gaussian_kde(y_test[:,k])(np.linspace(0,1,200)), label = k)
    ax.set_title(str(k))
plt.savefig('actual_class_probs.png')
plt.show()

