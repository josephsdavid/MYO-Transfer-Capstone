import json
import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


batch=128
clr=cb.OneCycleLR(
                 max_lr=.01,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


optim = SGD(momentum=0.9, nesterov=True)

strategy = tf.distribute.MirroredStrategy()

def holder(a,b):
	return a, b

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch, scale=False
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = batch, scale=False)


with strategy.scope():
    inputs = Input((52,8))
    x = inputs
    x = LSTM(8, recurrent_dropout=0.1, dropout=0.1)(x)
    outputs = Dense(7, activation='softmax')(x)
    source_model = Model(inputs, outputs)
    source_model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics = ['accuracy'])

source_model.fit(train_set, epochs=20, validation_data=val_set, callbacks=[clr], use_multiprocessing=True, workers=28, shuffle = False)
