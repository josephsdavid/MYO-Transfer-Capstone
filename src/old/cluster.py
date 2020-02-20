import n2d
import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

batch=1000

train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, batch_size=100, scale = True, sample_0=False, rectify=True)
test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=True, batch_size=100, scale = True, sample_0=False, rectify=True)

outs = [[] for i in range(4)]
for tr in train:
    outs[0].append(tr[0])
    outs[1].append(tr[1])
# augmentation is joyous
for tr, te in zip(train, test):
    outs[0].append(tr[0])
    outs[1].append(tr[1])
    outs[2].append(te[0])
    outs[3].append(te[1])

x_tr, y_tr, x_val, y_val = (np.concatenate(o, 0) for o in outs)

n_clusters=18

latent_dim=n_clusters
inputs = Input(shape=(52, 8))
o = LSTM(40, activation = 'tanh', return_sequences=True)(inputs)
o2 = LSTM(20, activation = 'tanh', return_sequences=True)(o)
encoder = LSTM(latent_dim, return_state=True, activation='tanh')
encoder_outputs, state_h, state_c = encoder(o2)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
decoded = RepeatVector(52)(encoder_outputs)
# Set up the decoder, using `encoder_states` as initial state.
decoder_lstm = LSTM(latent_dim, return_sequences=True, activation='tanh')
decoder_outputs = decoder_lstm(decoded, initial_state=encoder_states)
decoder_outputs =LSTM(20, activation='tanh', return_sequences=True)(decoder_outputs)
decoder_outputs =LSTM(40, activation='tanh', return_sequences=True)(decoder_outputs)
decoder_dense = TimeDistributed(Dense(8, activation='sigmoid'))
decoder_outputs = decoder_dense(decoder_outputs)

ae = n2d.autoencoder_generator((inputs, encoder_outputs, decoder_outputs))

umap=n2d.UmapGMM(n_clusters, umap_dim=18, umap_neighbors=20)

nd = n2d.n2d(ae, umap)
nd.fit(x_tr, batch_size=batch, epochs=1000, patience=400, weight_id="n2d.h5")

pred = nd.predict(x_tr)

nd.assess(y_tr)

nd.visualize(y_tr, None, 18)
