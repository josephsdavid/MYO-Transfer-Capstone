import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


strategy = tf.distribute.MirroredStrategy()
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


epochs = len(train_set)*4

for e in range(epochs):
    print("epoch: {}".format(e+1))
    lstm.fit(train_set, steps_per_epoch=1, validation_data=val_set, use_multiprocessing=True, epochs=epochs, shuffle=False, val_steps=1)
    lstm.reset_states()

lstm.save("result/stateful_lstm.h5")
