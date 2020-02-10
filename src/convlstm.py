#%%
#%%
%load_ext autoreload
%autoreload 2

#%%
import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Dropout
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
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
batch = 800

s = False
train = u.NinaGeneratorConv("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch, scale = True, sample_0=True)

test = u.NinaGeneratorConv("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        None, validation=True, by_subject = s, batch_size=batch, scale = True, sample_0=False)



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

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=32, kernel_size=(1, 5),data_format='channels_last',
                   input_shape=(None, 1, 26, 8),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),data_format='channels_last',
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),data_format='channels_last',
                   padding='same', return_sequences=False))
seq.add(Flatten())
seq.add(Dense(512, activation='relu'))
seq.add(Dropout(0.5))
seq.add(Dense(18, activation='softmax'))
seq.compile(loss='sparse_categorical_crossentropy', 
            optimizer=optim, metrics=['accuracy'])

#%%
seq.summary()


# %%
history = seq.fit(train, epochs=50, callbacks=[clr ],
        validation_data=test, shuffle = False)

# %%# %%
