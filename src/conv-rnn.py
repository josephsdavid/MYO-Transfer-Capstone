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
batch = 64

s = True
train = u.NinaGeneratorConv("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch, scale = False, shape_option=2)

test = u.NinaGeneratorConv("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        None, validation=True, by_subject = s, batch_size=batch, scale = False, shape_option=2)



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


model = Sequential()
model.add(TimeDistributed(Conv1D(filters=20, kernel_size=7, activation='relu'), input_shape=(None,26,8)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(52,dropout=.2, return_sequences=True))
model.add(LSTM(52, dropout=.2,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(18, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
            optimizer='adam', metrics=['accuracy'])
#%%
model.summary()


# %%
history = model.fit(train, epochs=50, #callbacks=[clr ],
        validation_data=test, shuffle = False)

# %%# %%
