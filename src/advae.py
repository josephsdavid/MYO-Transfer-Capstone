import tensorflow as tf
from PyEMD import EMD
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

batch = 3000

strategy = tf.distribute.MirroredStrategy()
clr=cb.OneCycleLR(
                 max_lr=0.4,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
       None, validation=False, by_subject = True, batch_size=batch)
test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
       None, validation=True, by_subject = True, batch_size=batch)




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

x_tr = np.concatenate(x_tr,0).astype(np.float16)
y_tr = np.concatenate(y_tr,0).astype(np.float16)
x_val = np.concatenate(x_val,0).astype(np.float16)
y_val = np.concatenate(y_val,0).astype(np.float16)

x_flat = np.array([x_tr[i,:,:].flatten('F') for i in range(x_tr.shape[0])])
v_flat = np.array([x_val[i,:,:].flatten('F') for i in range(x_val.shape[0])])
del(x_tr)
del(x_val)
del(train)
del(test)


def dense(x, ni, no, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[ni,no],
                initializer=tf.random_normal_initializer(mean=0., stddev=0.01))

        bias=tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out =  tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def encoder(x, reuse=False, layers=[1000, 1000], latent_dim=2):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, x.shape[-1], layers[0], 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, layers[0], layers[1], 'e_dense_2'))
        latent_variable=dense(e_dense_2, layers[1], latent_dim, 'e_latent_variable')
        return latent_variable



with strategy.scope():
    inputs = Input(shape=(52, 8))
    x = inputs
    drop=0.028638817753399493
    rec_drop = 0.14330185842693177
    x1 = LSTM(450, activation="tanh", dropout=drop, recurrent_dropout=rec_drop, return_sequences=True, name = "lstm_1")(x)
    x2 = LSTM(150, activation="tanh", dropout=drop, recurrent_dropout=rec_drop, return_sequences=True, name = "lstm_2")(x1)
    x3 = LSTM(400, activation="tanh", dropout=drop, recurrent_dropout=rec_drop, return_sequences=False, name = "lstm_3")(x2)

    outputs = Dense(150)(x3)
    outputs = Dense(17, activation='softmax', name = "gesture")(outputs)

    out2 = TimeDistributed(Dense(16), name = "fft")(x2)
    # make first encoder layer
   # o, h0, c0  = LSTM(400, activation = 'tanh', return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(inputs)
   # # make second encoder layer
   # o2, h1, c1 = LSTM(200, activation = 'tanh', return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(o)
   # # inner encoder layer
   # encoder = LSTM(latent_dim, return_state=True, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
   # encoder_outputs, state_h, state_c = encoder(o2)
   # # We discard `encoder_outputs` and only keep the states.
   # encoder_states = [state_h, state_c]
   # decoded = RepeatVector(52)(encoder_outputs)

   # classif = LSTM(latent_dim, activation='tanh', dropout=0.5, recurrent_dropout=0.5)(decoded, initial_state=encoder_states)
   # classif = Dense(17, activation='softmax', name = 'classification')(classif)
   # # Set up the decoder, using `encoder_states` as initial state.
   # decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
   # decoder_outputs, _, _ = decoder_lstm(decoded,
   # 		                                     initial_state=encoder_states)
   # decoder_outputs =LSTM(200, activation='tanh', return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(decoder_outputs, initial_state=[h1,c1])
   # decoder_outputs =LSTM(400, activation='tanh', return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(decoder_outputs, initial_state=[h0,c0])
   # decoder_dense = TimeDistributed(Dense(8), name='autoencoder')
   # decoder_outputs = decoder_dense(decoder_outputs)

    multi_model=Model(inputs, [outputs, out2])
    optim = SGD(momentum=0.9, nesterov=True)
    multi_model.compile(optimizer=optim,
                        loss = {'gesture':'sparse_categorical_crossentropy',
                                'fft':'mean_squared_error'}, metrics = {'gesture':'accuracy'})

multi_model.fit(x_tr,
                {'gesture':y_tr, 'fft':y_ft}, shuffle=False,
                epochs=100, validation_data = (x_val, {'gesture':y_val, 'fft':f_val}), batch_size=batch, callbacks=[clr, EarlyStopping(monitor='val_gesture_loss', patience=25)])


model_json = multi_model.to_json()

with open("result/multi.json", "w") as json_file:
        json_file.write(model_json)

multi_model.save_weights("result/multi.h5")
print("Saved model to disk")
