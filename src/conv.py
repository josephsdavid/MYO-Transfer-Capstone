#%%
#%%
%load_ext autoreload
%autoreload 2

#%%
import tensorflow as tf
import numpy as np
import utils as u
from utils import NinaGenerator
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dropout, Conv1D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

## Weird CUDA things... 
## https://github.com/tensorflow/tensorflow/issues/24496
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
## END CUDA FIX
#%%
batch = 200

s = False
scale = False
train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch, scale = scale,rectify=True)

test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        None, validation=True, by_subject = s, batch_size=batch, scale = scale,rectify=True)

#%%



#%%
clr=cb.OneCycleLR(
                 max_lr=0.4,
                 end_percentage=0.2,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)
optim = SGD(momentum=0.9, nesterov=True)
#%%


# inputs = Input((52,8))
# c_layer = Conv1D(filters=64, kernel_size=7, activation='relu')(inputs)
# c_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(c_layer)
# c_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(c_layer)
# c_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(c_layer)
# c_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(c_layer)
# do1 = Dropout(0.5)(c_layer)
# mp1 = MaxPooling1D(pool_size=2)(do1)
# fl1 = Flatten()(mp1)
# dense1 = Dense(100, activation='relu')(fl1)
# do2 = Dropout(0.5)(dense1)
# outputs = Dense(18, activation='softmax')(do2)
# model = Model(inputs, outputs)
# model.compile(loss='sparse_categorical_crossentropy', 
#             optimizer='adam', metrics=['accuracy'])

# #%%
# model.summary()


# # %%
# history = model.fit(train, epochs=50, #callbacks=[clr ],
#         shuffle = False)

# %%

def evaluate_model(_train, _test):
    verbose, epochs = 1, 15
    inputs = Input((52,8))
    c_layer = Conv1D(filters=32, kernel_size=1, activation='relu')(inputs)
    c_layer = Conv1D(filters=32, kernel_size=1, activation='relu')(c_layer)
    c_layer = Conv1D(filters=32, kernel_size=1, activation='relu')(c_layer)
    c_layer = MaxPooling1D(pool_size=2)(c_layer)
    c_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(c_layer)
    c_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(c_layer)
    c_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(c_layer)
    # c_layer = MaxPooling1D(pool_size=2)(c_layer)
    # c_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(c_layer)
    # c_layer = MaxPooling1D(pool_size=2)(c_layer)
    # c_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(c_layer)
    mp1 = MaxPooling1D(pool_size=2)(c_layer)
    fl1 = Flatten()(mp1)
    dense1 = Dense(128, activation='relu')(fl1)
    do2 = Dropout(0.5)(dense1)
    outputs = Dense(18, activation='softmax')(do2)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy'])
    model.fit(_train, epochs=epochs, #callbacks=[clr ],
        shuffle = False, validation_data=_test, verbose=verbose)
    _, accuracy = model.evaluate(_test, verbose=verbose)
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(train, test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment


# %%
run_experiment()

 # %%
