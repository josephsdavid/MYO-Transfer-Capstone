import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
# load up some data

batch=300

train = u.NinaGenerator("../data/ninaPro", ["c"],
                        process_fns=[u.butter_highpass_filter],
                        augment_fns=[u.add_noise_snr],
                        validation=False, by_subject = True, batch_size=batch)

test = u.NinaGenerator("../data/ninaPro", ["c"], [u.butter_highpass_filter],
                       None, validation=True, by_subject = True, batch_size=batch)

"""
generators are the move for making your code faster, they load the data one batch at
a time and tensorflow uses them in hella parallel, vs slow af numpy arrays
"""

print(train[0][0].shape)
#(300, 52, 8)
print(train[0][1].shape)
#(300,)

print(len(train))
# 693 = n_samples in our entire training set dividided by batch size
print(train.labels.shape[0]/batch)
# 693
# the advantage is it also can augment data as we go.
# but here lets populate a numpy array real quick:

x_tr = []
y_tr = []
for t in train:
    x_tr.append(t[0])
    y_tr.append(t[1])

x_val = []
y_val = []
for t in test:
    x_val.append(t[0])
    y_val.append(t[1])

sets = [x_tr, y_tr, x_val, y_val]
x_tr, y_tr, x_val, y_val = (np.concatenate(s, 0).astype(np.float16) for s in sets)

def maybefft(arr):
    a2 = np.fft.fft(arr)
    return np.concatenate((a2.imag, a2.real), axis=-1)

x_tr.shape
# (207900, 52, 8) = chefs kiss

y_fft = maybefft(x_tr)
v_fft = maybefft(x_val)

y_fft.shape
# (207900, 52, 16)

inputs = Input((52, 8))

lstm_1 = LSTM(450, dropout = 0.02, recurrent_dropout = 0.14, return_sequences=True)
lstm_1_out = lstm_1(inputs)
lstm_2_out = LSTM(150, dropout = 0.02, recurrent_dropout = 0.14, return_sequences=True)(lstm_1_out)
# now we split the boy up

# the fft network
fft_1 = TimeDistributed(Dense(150))(lstm_2_out)
fft_out = TimeDistributed(Dense(16), name = "fft")(fft_1)

# the classifier
classif_1 = LSTM(400, dropout=0.02, recurrent_dropout=0.14)(lstm_2_out)
classif_out = Dense(17, activation="softmax", name = "classif")(classif_1)

# build the jesse
fancyboy = Model(inputs, [fft_out, classif_out])

fancyboy.compile(optimizer="adam", loss = {"classif":"sparse_categorical_crossentropy", "fft":"mean_squared_error"})

trainY = {"classif":y_tr, "fft":y_fft}
valY = {"classif":y_val, "fft":v_fft}

fancyboy.fit(x_tr, trainY, epochs=100, validation_data = (x_val, valY), batch_size=batch)

