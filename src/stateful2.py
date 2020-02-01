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


batch=20


train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
                                [u.add_noise], batch_size = batch,
                                step=1, window_size=30, shuffle=False)
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
                            [u.add_noise], batch_size = batch,
                            step=1, window_size=30, shuffle=False)

inputs = Input(batch_shape=(batch, 30, 8))
x = LSTM(300, activation = 'tanh',
		dropout=0.5, recurrent_dropout=0.5, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
check = ModelCheckpoint("result/stateful_lstm.h5", monitor="val_loss", save_best_only=True)


stopper = EarlyStopping(monitor="val_loss", patience=20)


for e in range(len(train_set)*4):
	print("epoch: {}/{}".format(e, len(train_set)*4))
	lstm.fit(train_set, steps_per_epoch=48, epochs=1, verbose=0, workers = 12, use_multiprocessing=True, shuffle = False)
	lstm.reset_states()
	if (e%(len(train_set)//10)==0 and e !=0):
		lstm.evaluate(val_set, workers=12, use_multiprocessing=True, steps=len(val_set))
		lstm.save("result/stateful_lstm.h5")

lstm.save("result/stateful_lstm.h5")
