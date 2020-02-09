#%%
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

## Weird CUDA things... 
## https://github.com/tensorflow/tensorflow/issues/24496
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
## END CUDA FIX
#%%
train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, batch_size=100, scale = True)
x_tr = []
y_tr = []
for t in train:
    x_tr.append(t[0])
    y_tr.append(t[1])

x_tr = np.abs(np.concatenate(x_tr,0)).astype(np.float16)
y_tr = np.concatenate(y_tr,0).astype(np.float16)


#%%
latent_dim=100
inputs = Input(shape=(52, 8))
o = LSTM(400, activation = 'tanh', return_sequences=True)(inputs)
o2 = LSTM(200, activation = 'tanh', return_sequences=True)(o)
encoder = LSTM(latent_dim, return_state=True, activation='tanh')
encoder_outputs, state_h, state_c = encoder(o2)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
decoded = RepeatVector(52)(encoder_outputs)
# Set up the decoder, using `encoder_states` as initial state.
decoder_lstm = LSTM(latent_dim, return_sequences=True, activation='tanh')
decoder_outputs = decoder_lstm(decoded, initial_state=encoder_states)
decoder_outputs =LSTM(200, activation='tanh', return_sequences=True)(decoder_outputs)
decoder_outputs =LSTM(400, activation='tanh', return_sequences=True)(decoder_outputs)
decoder_dense = TimeDistributed(Dense(8, activation='sigmoid'))
decoder_outputs = decoder_dense(decoder_outputs)
ae = Model(inputs, decoder_outputs)
ae.compile('adam', 'mse')
ae.summary();
ae.fit(x_tr, x_tr, batch_size=100, shuffle=False, epochs = 100)

#%%

# Encoder
# lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
# lstm_autoencoder.add(RepeatVector(timesteps))
# # Decoder
# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
# lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
# lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

# lstm_autoencoder.summary()

# #%%
# test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
#         None, validation=True, by_subject = s, batch_size=batch, scale = False)
# sub = 'subject' if s else 'repetition'
# loc = 'notransfer_'+sub+'_'+abc[i]
# out_shape = to_categorical(train.labels).shape[-1]
# print('beginning ' +loc )
# inputs = Input((52,8))
# x, s1, s2, s3, s4 = Bidirectional(LSTM(64, dropout = drop, recurrent_dropout=rec_drop, return_sequences=True, return_state=True))(inputs)
# x = PReLU()(x)
# x = Bidirectional(LSTM(64, dropout = drop, recurrent_dropout=rec_drop, return_sequences=False))(x, initial_state = [s1,s2,s3,s4])
# x = PReLU()(x)
# outputs = Dense(out_shape, activation='softmax')(x)
# lstm = Model(inputs, outputs)
# lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history = lstm.fit(train, epochs=50,
#         callbacks=[EarlyStopping(patience=100, monitor='val_loss'), clr ],
#         validation_data=test, shuffle = False, workers=12, use_multiprocessing=True)
# results[loc] = lstm.evaluate(test)
# lstm.save("models/lstm_{}.h5".format(loc))
# plt.subplot(212)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # summarize history for loss
# plt.subplot(211)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# F = plt.gcf()
# Size = F.get_size_inches()
# F.set_size_inches(Size[0]*2, Size[1]*2)
# plt.savefig("training_{}.png".format(loc))
# del(history)
# print(results)

# import json
# with open('result.json', 'w') as fp:
#     json.dump(results, fp)


# %%
