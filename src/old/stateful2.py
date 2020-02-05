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


batch=3


train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
                                [u.add_noise], batch_size = batch,
                                step=1, window_size=49, shuffle=False)
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
                            [u.add_noise], batch_size = batch,
                            step=1, window_size=49, shuffle=False)


dropout = 0.028638817753399493
rec_drop=0.14330185842693177
inputs = Input(batch_shape=(batch, 49, 8))
x = LSTM(20, activation = 'tanh',
		dropout=dropout, recurrent_dropout=rec_drop, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)

lstm.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics= ['accuracy'])


for e in range(100000):
	print("/nepoch: {}/{}".format(e, 10000))
	lstm.fit(train_set, steps_per_epoch=len(train_set)//100, epochs=1, verbose=1, workers = 12, use_multiprocessing=True, shuffle = False)
	lstm.reset_states()
	if (e%10000==0 and e !=0):
		lstm.evaluate(val_set, workers=12, use_multiprocessing=True, steps=len(val_set)//10)
		lstm.save("result/stateful_lstm.h5")

lstm.save("result/stateful_lstm_good.h5")
