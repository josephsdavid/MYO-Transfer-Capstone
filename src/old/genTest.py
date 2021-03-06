import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

x =( 'ok')

print("beginning initial training")

batch=3000

test_res = {}


optim = SGD(momentum=0.9, nesterov=True)

strategy = tf.distribute.MirroredStrategy()

def holder(a,b):
    return a, b

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch, scale=False
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = batch, scale=False)


with strategy.scope():
    inputs = Input((52,8))
    x = inputs
    drop=0.028638817753399493
    rec_drop = 0.14330185842693177
    units=[450, 150, 400]
    seq=[True, True, False]
    clr=cb.OneCycleLR(max_lr=0.4,end_percentage=0.2,scale_percentage=None,maximum_momentum=0.95,minimum_momentum=0.85,verbose=True)
    for i in range(3):
        x = LSTM(units[i], activation="tanh", dropout=0.5, recurrent_dropout=0.5, return_sequences=seq[i], name = "lstm_{}".format(i))(x)
        outputs = Dense(7, activation='softmax')(x)
        source_model = Model(inputs, outputs)
        optim = SGD(momentum=0.9, nesterov=True)
        source_model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics = ['accuracy'])
        #	# https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

print(source_model.summary())
stopper = EarlyStopping(monitor = "val_loss", patience=1000)
history = source_model.fit(train_set, epochs=150, validation_data=val_set, callbacks=[stopper, clr], workers=16, use_multiprocessing=True, steps_per_epoch=len(train_set)//4, shuffle = False)

abc = ['a','b','c']
subject=[False, True]

results = {}


for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=0.4,
            end_percentage=0.2,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch)
        sub = 'subject' if s else 'repetition'
        loc = 'notransfer_'+sub+'_'+abc[i]
        out_shape = to_categorical(train.labels).shape[-1]
        print('beginning ' +loc )
        with strategy.scope():
            inputs = Input((52,8))
            x = inputs
            drop=0.028638817753399493
            rec_drop = 0.14330185842693177
            units=[450, 150, 400]
            seq=[True, True, False]
            for i in range(3):
                x = LSTM(units[i], activation="tanh", dropout=drop, recurrent_dropout=rec_drop, return_sequences=seq[i], name = "lstm_{}".format(i))(x)

            outputs = Dense(150)(x)
            outputs = Dense(out_shape, activation='softmax')(outputs)
            lstm = Model(inputs, outputs)
            lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        lstm.fit(train, epochs=150,
                 callbacks=[clr, EarlyStopping(patience=1000, monitor='val_loss')],
                 validation_data=test, shuffle = False)
        results[loc] = lstm.evaluate(test)



with open("result/transfer_results_no_shuffle1.txt","w") as f:
    f.write(str(results))
    f.close()

for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=0.4,
            end_percentage=0.2,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch)
        out_shape = to_categorical(train.labels).shape[-1]
        with strategy.scope():
            transfer_model = tf.keras.models.clone_model(source_model)
            hidden = Dense(120)(transfer_model.layers[-2].output)
            out=Dense(out_shape, activation='softmax')(hidden)
            lstm = Model(transfer_model.input, out)
            lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            sub = 'subject' if s else 'repetition'
            loc = 'unfrozen_'+sub+'_'+abc[i]
            print('beginning '+ loc)

        lstm.fit(train, epochs=150,
                 callbacks=[clr, EarlyStopping(patience=1000, monitor='val_loss')],
                 validation_data=test, shuffle = False)
        results[loc] = lstm.evaluate(test)



with open("result/transfer_results_no_shuffle2.txt","w") as f:
    f.write(str(results))
    f.close()



for i in range(len(abc)):
    for s in subject:
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch)
        out_shape = to_categorical(train.labels).shape[-1]
        with strategy.scope():
            transfer_model = tf.keras.models.clone_model(source_model)
            transfer_model.trainable=True
            hidden = Dense(120)(transfer_model.layers[-2].output)
            out=Dense(out.shape, activation='softmax')(hidden)
            lstm = Model(transfer_model.input, out)
            lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            sub = 'subject' if s else 'repetition'
            loc = 'frozen_'+sub+'_'+abc[i]
            print('beginning ' + loc)
        lstm.fit(train, epochs=150,
                 callbacks=[clr, EarlyStopping(patience=1000, monitor='val_loss')],
                 validation_data=test, shuffle=False)
        results[loc] = lstm.evaluate(test)



with open("result/transfer_results_no_shuffle3.txt","w") as f:
    f.write(str(results))
    f.close()

