# %%
# %%
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Conv3D, Dropout, Conv1D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU, RepeatVector, TimeDistributed, Flatten
import callbacks as cb
from utils import NinaGenerator
import utils as u
import numpy as np
import tensorflow as tf
%load_ext autoreload
%autoreload 2


# Weird CUDA things...
# https://github.com/tensorflow/tensorflow/issues/24496

# %%
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# END CUDA FIX
# %%
batch = 64
s = False
scale = False
rectify = True
shape_option = 3
train = u.NinaGeneratorConv("../data/ninaPro",
            ['b'],
            [u.butter_highpass_filter],
            [u.add_noise_snr],
            validation=False,
            by_subject=s,
            batch_size=batch,
            scale=scale,
            rectify=rectify,
            shape_option=shape_option)

test = u.NinaGeneratorConv("../data/ninaPro",
            ['b'],
            [u.butter_highpass_filter],
            None,
            validation=True,
            by_subject=s,
            batch_size=batch,
            scale=scale,
            rectify=rectify,
            shape_option=shape_option)


# %%
clr = cb.OneCycleLR(
                 max_lr=0.4,
                 end_percentage=0.2,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)
optim = SGD(momentum=0.9, nesterov=True)
# %%
# build model


def build_model():
    inputs = Input((1,52,8))
    c_layer = Conv2D(filters=64,padding='same', kernel_size=(2,3), activation='relu')(inputs)
    c_layer = Conv2D(filters=64,padding='same', kernel_size=(2,3), activation='relu')(c_layer)
    c_layer = Conv2D(filters=64,padding='same', kernel_size=(2,3), activation='relu')(c_layer)
    c_layer = MaxPooling2D(pool_size=(1,2))(c_layer)
    c_layer = Conv2D(filters=128,padding='same', kernel_size=(2,3), activation='relu')(c_layer)
    c_layer = Conv2D(filters=128,padding='same', kernel_size=(2,3), activation='relu')(c_layer)
    c_layer = Conv2D(filters=128,padding='same', kernel_size=(2,3), activation='relu')(c_layer)
    mp1 = MaxPooling2D(pool_size=(1,2))(c_layer)
    fl1 = Flatten()(mp1)
    dense1 = Dense(512, activation='relu')(fl1)
    do2 = Dropout(0.5)(dense1)
    outputs = Dense(18, activation='softmax')(do2)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    
    return model
# %%


def evaluate_model(_train, _test, _model):
    verbose, epochs = 1, 25
    _model.fit(_train, epochs=epochs,  # callbacks=[clr ],
        shuffle=False, validation_data=_test, verbose=verbose)
    _, accuracy = _model.evaluate(_test, verbose=verbose)
    return accuracy

# summarize scores


def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment

def run_experiment(repeats=10):

    scores = list()
    for r in range(repeats):
        mod = build_model()
        score = evaluate_model(train, test, mod)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)

    return summarize_results(scores)
 
# run the experiment


# %%
run_experiment()

 # %%
