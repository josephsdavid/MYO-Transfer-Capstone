import numpy as np
from dataloaders import train_loader, val_loader, test_loader
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, LSTM

X_train, y_train = train_loader("../EvaluationDataset")
y_train = to_categorical(y_train)

X_val, y_val = val_loader("../EvaluationDataset")
y_val = to_categorical(y_val)

X_test, y_test = test_loader("../EvaluationDataset")
y_test = to_categorical(y_test)

scores = {}

scores['simple_lstm_no_training'] = load_model("models/simple_lstm.h5").evaluate(X_test, y_test)
scores['wide_lstm_no_training'] = load_model("models/wide_lstm.h5").evaluate(X_test, y_test)

simple_source = load_model("models/simple_lstm.h5")
simple_source.trainable= False

hidden = Dense(120, activation = 'relu')(simple_source.layers[-2].output)
out = Dense(7, activation='softmax')(hidden)

model_transfer = Model(simple_source.input, out)

model_transfer.summary()

comp_opts = {
    'loss' : 'categorical_crossentropy',
    'metrics' : ['accuracy']}

Adam = tf.keras.optimizers.Adam(lr = 0.001)
model_transfer.compile(optimizer = Adam, **comp_opts)

model_transfer.fit(X_train, y_train, batch_size = 400, epochs = 25, validation_data = (X_val, y_val))

scores['simple_lstm_transfer'] = model_transfer.evaluate(X_test, y_test)


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
        Adam = tf.keras.optimizers.Adam(lr = lr)
        self.model.compile(optimizer = Adam, **compilation_options)
        self.history = self.model.fit(X, y, **fit_options)


lstm = simple_lstm_classifier(X_train, y_train, dropout = 0.1)

fit_options = {
    'epochs': 25,
    'batch_size':400,
    'shuffle':True,
    'verbose':1,
    'validation_data':(X_val, y_val)
}

lstm.fit(X_train, y_train, lr = 0.005, fit_options=fit_options)

scores['lstm_trained_simple'] = lstm.model.evaluate(X_test, y_test)



wide_source = load_model("models/wide_lstm.h5")
wide_source.trainable= False

hidden = Dense(120, activation = 'relu')(wide_source.layers[-2].output)
out = Dense(7, activation='softmax')(hidden)

model_transfer = Model(wide_source.input, out)

model_transfer.summary()

comp_opts = {
    'loss' : 'categorical_crossentropy',
    'metrics' : ['accuracy']}

Adam = tf.keras.optimizers.Adam(lr = 0.001)
model_transfer.compile(optimizer = Adam, **comp_opts)

model_transfer.fit(X_train, y_train, batch_size = 400, epochs = 25, validation_data = (X_val, y_val))

scores['wide_lstm_transfer-frozen'] = model_transfer.evaluate(X_test, y_test)


simple_source = load_model("models/simple_lstm.h5")

hidden = Dense(120, activation = 'relu')(simple_source.layers[-2].output)
out = Dense(7, activation='softmax')(hidden)

model_transfer = Model(simple_source.input, out)

model_transfer.summary()

comp_opts = {
    'loss' : 'categorical_crossentropy',
    'metrics' : ['accuracy']}

Adam = tf.keras.optimizers.Adam(lr = 0.001)
model_transfer.compile(optimizer = Adam, **comp_opts)

model_transfer.fit(X_train, y_train, batch_size = 400, epochs = 25, validation_data = (X_val, y_val))

scores['simple_lstm_transfer-unfrozen'] = model_transfer.evaluate(X_test, y_test)

import json

scores2 = {k: [float(x) for x in scores[k]] for k in scores.keys()}

with open('scores.json', 'w') as outfile:
    json.dump(scores2, outfile)


