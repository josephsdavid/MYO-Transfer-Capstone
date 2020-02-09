import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU, Flatten, BatchNormalization, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

batch=400


def noise(snr):
    def res(x):
        return u.add_noise_snr(x, snr)
    return res

nnoise = noise(5)
# or


optim = SGD(momentum=0.9, nesterov=True)
#optim = Nadam(learning_rate=0.005)


abc = ['b','a','c']
subject=[True, False]
rec_drop = 0.5
drop=0.5

results = {}
hl = []

for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=1,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch,
                scale = False, rectify=True)
        print(train[0][1])
        print(np.unique(train.subject))
        print(np.unique(train.rep))
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                None, validation=True, by_subject = s, batch_size=batch,
                scale = False, rectify = True)
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
        x_tr, y_tr, x_val, y_val = (np.concatenate(x,axis=0) for x in [x_tr, y_tr, x_val, y_val])
        #x_tr /= 128
        #x_val /=128
        sub = 'subject' if s else 'repetition'
        loc = 'notransfer_'+sub+'_'+abc[i]
        out_shape = to_categorical(train.labels).shape[-1]
        print('beginning ' +loc )
        lstm_size=64
        inputs = Input((52,8))
        encoded = Bidirectional(LSTM(lstm_size*2, dropout=drop, recurrent_dropout=drop, return_sequences=True))(inputs)
        encoded2, h1, b1, h2, b2 = Bidirectional(LSTM(lstm_size, dropout=drop, recurrent_dropout=rec_drop, return_state=True, return_sequences=False))(encoded)
        decoded = RepeatVector(52)(encoded2)
        decoded = Bidirectional(LSTM(lstm_size, dropout=drop, recurrent_dropout=rec_drop, return_state=False, return_sequences=True))(decoded, initial_state=[h1,b1,h2,b2])
        encoded = Bidirectional(LSTM(lstm_size*2, dropout=drop, recurrent_dropout=drop, return_sequences=True))(decoded)
        dec_out = TimeDistributed(Dense(8), name = "decoder")(decoded)
        #x, s1, s2, s3, s4 = Bidirectional(LSTM(lstm_size, dropout = drop, recurrent_dropout=rec_drop, return_sequences=False, return_state=True, bias_initializer='ones'))(inputs, initial_state = [h1,b1,h2,b2])
        #x = PReLU()(x)
        x = Dense(500)(encoded2)
        x = PReLU()(x)
        x = Dense(1000)(x)
        x = PReLU()(x)
        x = Dense(1000)(x)
        x = PReLU()(x)
        #x = Bidirectional(LSTM(lstm_size, dropout = drop, recurrent_dropout=rec_drop, return_sequences=False))(x, initial_state = [s1,s2,s3,s4])
        #x = PReLU()(x)
        outputs = Dense(out_shape, activation='softmax', name="classifier")(x)
        lstm = Model(inputs, [outputs, dec_out])
        #if s:
        #    plot_model(lstm, show_shapes=True, expand_nested=True, to_file="model.png")

        lstm.compile(optimizer=SGD(learning_rate=1e-2, momentum=0.9, nesterov=True, decay=1e-4),
                     loss={'classifier':'sparse_categorical_crossentropy','decoder':'mse'}, metrics=['accuracy'])

        cb = [EarlyStopping(patience=10, monitor='val_classifier_loss'), ModelCheckpoint("models/lstm_{}.h5".format(loc),monitor='val_classifier_loss', save_best_only=True)]

        history = lstm.fit(x_tr, {'decoder':x_tr, 'classifier':y_tr},
                           epochs=500, batch_size=batch,
                           validation_data=(x_val, {'decoder':x_val, 'classifier':y_val}),
                           shuffle = False, workers=12, use_multiprocessing=True)
        hl.append(history)

        results[loc] = lstm.evaluate(x_val)
        #lstm.save("models/lstm_{}.h5".format(loc))
        #plt.subplot(212)
        #plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        ## summarize history for loss
        #plt.subplot(211)
        #plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        #plt.title('model loss')
        #plt.ylabel('loss')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #F = plt.gcf()
        #Size = F.get_size_inches()
        #F.set_size_inches(Size[0]*2, Size[1]*2)
        #plt.savefig("training_{}.png".format(loc))
        #del(history)
        #history = []
        print(results)

import json
with open('result.json', 'w') as fp:
    json.dump(results, fp)
