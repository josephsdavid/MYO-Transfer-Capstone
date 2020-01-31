import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
import kerastuner as kt
from tensorflow.keras.models import load_model
from scipy.stats import gaussian_kde
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.layers import Embedding, Activation, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical



batch=3000

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch)
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch)


def build_model(hp):
    n_lstms = hp.Int('lstm_layers', 1, 3, default = 1)
    ns = range(n_lstms)
    drop = hp.Float("dropout", 0, 0.5)
    rec_drop = hp.Float("recurrent_dropout", 0, 0.5)
    inputs = Input((52, 8))
    x = inputs
    for i in ns:
        seq = False if (i == (len(ns) -1)) else True
        x = LSTM(
            units=hp.Int('units_' + str(i), 100, 500, step = 50),
            dropout = drop,
            recurrent_dropout = rec_drop,
            return_sequences = seq, activation = 'tanh'
        )(x)

    outputs = Dense(7, activation = 'softmax')(x)

    model = Model(inputs, outputs)
    optim = SGD(lr=0.0025, momentum = 0.95, nesterov = True)
    model.compile(
        optimizer=optim,
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy']
    )
    return model

lr_manager = cb.OneCycleLR(1e-3,
    	end_percentage = 0.1, scale_percentage = None,
    	maximum_momentum = 0.95, minimum_momentum=0.85
    	)
stopper = EarlyStopping(monitor = "val_loss", patience=10)

tuner = kt.Hyperband(build_model, objective = 'val_accuracy', max_epochs=100, hyperband_iterations = 2, directory = 'result', project_name = 'lstm_tuning', distribution_strategy=tf.distribute.MirroredStrategy())


tuner.search(train_set, steps_per_epoch = int(len(train_set)/4), validation_data = val_set, validation_steps=int(len(val_set)/5), callbacks = [stopper, lr_manager])


