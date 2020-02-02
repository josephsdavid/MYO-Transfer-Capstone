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
        def on_train_batch_end(self, batch, logs={}):
                self.model.reset_states()

        def on_test_batch_end(self, batch, logs={}):
                self.model.reset_states()

batch=40
def holder(a,b):
	return a, b
train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
				[holder], batch_size = batch,
                                step=1, window_size=10, shuffle=False)
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
			    [holder], batch_size = batch,
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
x_tr = np.concatenate(x_tr,0).astype(np.float16)
y_tr = np.concatenate(y_tr,0).astype(np.float16)
x_val = np.concatenate(x_val,0).astype(np.float16)
y_val = np.concatenate(y_val,0).astype(np.float16)
print(x_tr.shape)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, targets, batch_size = 400, dim = (53), n_channels = 8, shuffle=True, n_classes=7):
        self.dim = dim
        self.batch_size = batch_size
        self.targets = targets
        self.features=features
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle =shuffle
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(self.features.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.indexes=np.arange(self.features.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.features[indexes,:,:],  self.targets[indexes]

t_set=DataGenerator(x_tr, y_tr, batch_size=batch, shuffle=False)
v_set=DataGenerator(x_val, y_val, batch_size=batch, shuffle=False)

dropout = 0.5
rec_drop=0.5

stopper = EarlyStopping(monitor = "val_loss", patience=10)
inputs = Input(batch_shape=(batch, 10, 8))
x = LSTM(20, activation = 'tanh',
        	dropout=dropout, recurrent_dropout=rec_drop, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics= ['accuracy'])

#epochs = 25
#for e in range(epochs):
#	print(e)
#	cnt=0
#	for i in (t_set):
#		print("sample: {}/{}".format(cnt, len(t_set)))
#		cnt +=1
#		lstm.train_on_batch(t[0], t[1])
#		lstm.reset_states()
#	lstm.evaluate(v_set)

lstm.fit(t_set, steps_per_epoch=len(t_set)//10, validation_data=v_set, validation_freq=10, epochs=100)
lstm.evaluate(x_val, y_val)
preds=lstm.predict(v_set)
lstm.save("result/stateful_lstm_attempted.h5")

from sklearn.metrics import accuracy_score

preds=np.mean(preds.reshape(-1,batch), axis=1).round()

res = []
for i in v_set:
    res.append(np.mean(i[1]))

len(res)

np.vstack(res).shape

print(accuracy_score(preds, np.hstack(res).T.round()))

