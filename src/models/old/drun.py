import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, PReLU, Flatten, BatchNormalization, RepeatVector, TimeDistributed, LSTM, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

batch=300


def noise(snr):
    def res(x):
        return u.add_noise_snr(x, snr)
    return res

nnoise = noise(1)
# or


optim = SGD(momentum=0.9, nesterov=True)
#optim = Nadam(learning_rate=0.005)


subject=[True, False]
rec_drop = 0.2
drop=0.2

results = {}
hl = []
abc = ['b','a','c']
for i in range(len(abc)):
    for s in subject:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        clr=cb.OneCycleLR(
            max_lr=0.01,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch,
                scale = True, rectify=False, sample_0=False)
        print(train[0][1])
        print(np.unique(train.subject))
        print(np.unique(train.rep))
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                None, validation=True, by_subject = s, batch_size=batch,
                scale = True, rectify = False, sample_0=False)


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
        y_val = np.hstack(y_val)
        y_tr = np.hstack(y_tr)
        x_tr, x_val = (np.concatenate(x,axis=0) for x in [x_tr, x_val])

        def or1(x, n):
            out =  1 if x==n else 0
            return out

        bin_tr=[]
        bin_val=[]
        for w in np.unique(y_tr):
            print(w)
            bin_tr.append(np.hstack([or1(y_tr[k],w) for k in range(y_tr.shape[0])]))
            bin_val.append(np.hstack([or1(y_val[k],w) for k in range(y_val.shape[0])]))


        vary = [str(x) for x in np.unique(y_tr)]
        train_dict = dict(zip(vary, bin_tr))
        val_dict = dict(zip(vary, bin_val))
        train_dict['classifier']=y_tr
        val_dict['classifier'] = y_val

        sub = 'subject' if s else 'repetition'
        loc = 'notransfer_'+sub+'_'+abc[i]
        out_shape = to_categorical(train.labels).shape[-1]
        print('beginning ' +loc )
        lstm_size=52
        inputs = Input((52,8))

        x, h1, b1, h2, b2 = Bidirectional(LSTM(lstm_size, dropout=drop, recurrent_dropout=drop, return_sequences=True, return_state=True))(inputs)

        x = Bidirectional(LSTM(lstm_size, dropout=drop, recurrent_dropout=rec_drop, return_state=False, return_sequences=False))(x, initial_state=[h1,b1,h2,b2])
        outlist = []
        for w in np.unique(y_tr):
            outlist.append(
                Dense(1, activation='sigmoid', name = str(w))(x)
            )

        output = Concatenate(name='classifier')(outlist)
        outlist.append(output)

        lstm = Model(inputs, outlist)
        losses = {str(w):'binary_crossentropy' for w in np.unique(y_tr)}
        losses['classifier'] = 'sparse_categorical_crossentropy'

        lstm.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9, nesterov=True, decay=1e-4),
                     loss=losses, metrics=['accuracy'])
        lstm.summary()
        plot_model(lstm, to_file="crazy.png", show_shapes=True, expand_nested = True)

        callb = [EarlyStopping(patience=1000, monitor='val_loss'), ModelCheckpoint("models/multi_{}.h5".format(loc),monitor='val_classifier_loss', save_best_only=True), clr]

        hl.append(lstm.fit(x_tr, train_dict, epochs=100,  callbacks=callb, validation_data=(x_val, val_dict),
                           shuffle=True,  batch_size=batch, verbose=2
                          # , steps_per_epoch=len(test)
                           ))


        results[loc] = lstm.evaluate(test)
        #lstm.save("models/lstm_{}.h5".format(loc))
        plt.subplot(212)
        plt.plot(hl[-1].history['accuracy'])
        plt.plot(hl[-1].history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.subplot(211)
        plt.plot(hl[-1].history['loss'])
        plt.plot(hl[-1].history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0]*2, Size[1]*2)
        plt.savefig("training_{}.png".format(loc))
        #del(history)
        #history = []
        print(results)

import json
with open('result.json', 'w') as fp:
    json.dump(results, fp)
