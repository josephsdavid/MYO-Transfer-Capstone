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

batch = 512

strategy = tf.distribute.MirroredStrategy()
clr=cb.OneCycleLR(
                 max_lr=0.01,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
       [u.add_noise_snr], validation=False, by_subject = True, batch_size=batch)
test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
       None, validation=True, by_subject = True, batch_size=batch)


with strategy.scope():
    inputs = Input(shape=(52, 8))
    # make first encoder layer
    o, h0, c0  = LSTM(400, activation = 'tanh', return_sequences=True, return_state=True)(inputs)
    # make second encoder layer
    o2, h1, c1 = LSTM(200, activation = 'tanh', return_sequences=True, return_state=True)(o)
    # inner encoder layer
    encoder = LSTM(latent_dim, return_state=True, activation='tanh')
    encoder_outputs, state_h, state_c = encoder(o2)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    decoded = RepeatVector(52)(encoder_outputs)

    classif = LSTM(latent_dem, activation='tanh')(decoded, initial_state=encoder_states)
    classif = Dense(17, activation='softmax')(classif)
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh')
    decoder_outputs, _, _ = decoder_lstm(decoded,
    		                                     initial_state=encoder_states)
    decoder_outputs =LSTM(200, activation='tanh', return_sequences=True)(decoder_outputs, initial_state=[h1,c1])
    decoder_outputs =LSTM(400, activation='tanh', return_sequences=True)(decoder_outputs, initial_state=[h0,c0])
    decoder_dense = TimeDistributed(Dense(8))
    decoder_outputs = decoder_dense(decoder_outputs)

    multi_model=Model(inputs, [classif, decoder_outputs])
    optim = SGD(momentum=0.9, nesterov=True)
    multi_model.compile(optimizer=optim,
                        loss = {'classification':'sparse_categorical_crossentropy',
                                'autoencoder':'mean_square_error'}, metrics = {'classification':'accuracy'})

multi_model.fit(train[0],
                {'classification':train[1], 'autoencoder':train[0]}, shuffle=False,
                epochs=100, validation_data = (test[0], {'classification':test[1], 'autoencoder':test[0]}),
                callbacks=[clr])
