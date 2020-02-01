import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
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
x = LSTM(40, activation = 'tanh',
         dropout=0.5, recurrent_dropout=0.5, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])


epochs = 100

for e in range(epochs):
    print("epoch: {}".format(e+1))
    history=lstm.fit(train_set, steps_per_epoch=len(train_set)//10,
                     validation_data=val_set, verbose=1, use_multiprocessing=True,
                     validation_steps=len(val_set)/5)
    print("loss: {}, accuracy: {}, val_loss: {}, val_acc: {}".format(
        history.history['loss'],
        history.history['accuracy'],
        history.history['val_loss'],
        history.history['val_accuracy']
    ))
    lstm.reset_states()

lstm.save("stateful_lstm")
