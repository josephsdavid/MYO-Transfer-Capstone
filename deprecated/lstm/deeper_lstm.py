import numpy as np
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Bidirectional
from tensorflow.keras.layers import Embedding, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import TensorBoard, Callback, EarlyStopping
from sklearn.metrics import roc_auc_score
import math
import matplotlib.pyplot as plt
from dataloaders import test_loader
from load_pretrain import  read_data_augmented


X, y = read_data_augmented("../PreTrainingDataset")
y = to_categorical(y)

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
            Adam = optimizers.Adam(lr = lr)
            self.model.compile(optimizer = Adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks])
        else:
            Adam = optimizers.Adam(lr = lr)
            self.model.compile(optimizer = Adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options)

lstm = stacked_lstm_classifier(X, y, dropout = 0.1)

lstm.fit(X,y, tensorboard = False)

history = lstm.history

plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
        x = (LSTM(120, activation = act,name = 'LSTM_1',
                  dropout = dropout,
                  recurrent_dropout = dropout
                  ))(start)
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
            Adam = optimizers.Adam(lr = lr)
            self.model.compile(optimizer = Adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options, callbacks = [callbacks])
        else:
            Adam = optimizers.Adam(lr = lr)
            callbacks = EarlyStopping(monitor = 'loss', patience = 10, mode = 'max')
            self.model.compile(optimizer = Adam, **compilation_options)
            self.history = self.model.fit(X, y, **fit_options)


lstm = wide_lstm_classifier(X, y, dropout = 0.1, batch_norm = False)


fit_options = {
    'epochs': 100,
    'batch_size':400,
    'shuffle':True,
    'validation_split':0.1,
    'verbose':1
}

lstm.fit(X,y, tensorboard=False, lr = 0.00005, fit_options = fit_options)

lstm.model.save("models/wide_lstm.h5")

history = lstm.history

print(history.history['val_accuracy'])


plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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



######## temporary evaluation
X_test, y_test = test_loader("../EvaluationDataset")
y_test = to_categorical(y_test)


score = lstm.model.evaluate(X_test, y_test)

print("%s: %.2f%%" % (lstm.model.metrics_names[1], score[1]*100))
71.79

preds = lstm.model.predict(X_test)

fig = plt.figure()
for k in range(preds.shape[-1]):
    ax = fig.add_subplot(3, 3, k+1)
    ax.plot(np.linspace(0,1, 200),gaussian_kde(preds[:,k])(np.linspace(0,1,200)), label = k)
    ax.set_title(str(k))
plt.savefig('wide_lstm_class_probs.png')
plt.show()

