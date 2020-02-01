import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


class ResetStatesCallback(Callback):
        def on_batch_end(self, batch, logs={}):
                self.model.reset_states()

batch=40
train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
                                [u.add_noise], batch_size = batch,
                                step=1, window_size=10, shuffle=False)
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
                            [u.add_noise], batch_size = batch,
                            step=1, window_size=10, shuffle=False)

x_tr = []
y_tr = []
for t in train_set:
    x_tr.append(t[0])
    y_tr.append(t[1])
x_val = []
y_val = []
for t in val_set:
    x_val.append(t[0])
    y_val.append(t[1])

import random
temp = list(zip(x_tr, y_tr))
random.shuffle(temp)
x_tr, y_tr = zip(*temp)
temp = list(zip(x_val, y_val))
random.shuffle(temp)
x_val, y_val = zip(*temp)
x_tr = np.concatenate(x_tr,0)
y_tr = np.concatenate(y_tr,0)
x_val = np.concatenate(x_val,0)
y_val = np.concatenate(y_val,0)

dropout = 0.5
rec_drop=0.5
inputs = Input(batch_shape=(batch, 10, 8))
x = LSTM(20, activation = 'tanh',
                dropout=dropout, recurrent_dropout=rec_drop, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics= ['accuracy'])


for e in range(x_tr.shape[0]):
        print("/nepoch: {}/{}".format(e, x_tr.shape[0]))
        lstm.train_on_batch(x_tr, y_tr)
        if e % 1000 ==0:
            lstm.reset_states()
            lstm.evaluate(x_val, y_val)

preds=lstm.predict(val_set)

lstm.save("result/stateful_lstm_good.h5")

from sklearn.metrics import accuracy_score

preds=np.mean(preds.reshape(-1,20), axis=1).round()

res = []
for i in val_set:
    res.append(np.mean(i[1]))

len(res)

np.vstack(res).shape

accuracy_score(preds, np.hstack(res).T.round())

