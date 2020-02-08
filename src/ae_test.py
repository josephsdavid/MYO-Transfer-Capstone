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
#%%
train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
        [u.add_noise_snr], validation=False, batch_size=batch, scale = False)
x_tr = []
y_tr = []
for t in train:
    x_tr.append(t[0])
    y_tr.append(t[1])

x_tr = np.concatenate(x_tr,0).astype(np.float16)
y_tr = np.concatenate(y_tr,0).astype(np.float16)


#%%
inputs_ae = Input(shape=(52, 8))
encoded_ae = LSTM(128, return_sequences=True, dropout=0.3)(inputs_ae, training=True)
encoded_ae = LSTM(32, return_sequences=False, dropout=0.3)(encoded_ae, training=True)
encoded = RepeatVector(52)(encoded_ae)
decoded_ae = LSTM(32, return_sequences=True, dropout=0.3)(encoded, training=True)
decoded_ae = LSTM(128, return_sequences=True, dropout=0.3)(decoded_ae, training=True)
out_ae = TimeDistributed(Dense(8))(decoded_ae)
sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
sequence_autoencoder.summary()

#%%
sequence_autoencoder.fit(x_tr, x_tr, batch_size=1000, epochs=100, verbose=2)

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
