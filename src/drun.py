import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, PReLU
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

batch=3000

clr=cb.OneCycleLR(
                 max_lr=0.5,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


optim = SGD(momentum=0.9, nesterov=True)


abc = ['a','b','c']
subject=[False, True]
rec_drop = 0.2
drop=0.2

results = {}

for i in range(len(abc)):
    for s in subject:
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch, scale = False)
        print(np.unique(train.subject))
        print(np.unique(train.rep))
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                None, validation=True, by_subject = s, batch_size=batch, scale = False)
        sub = 'subject' if s else 'repetition'
        loc = 'notransfer_'+sub+'_'+abc[i]
        out_shape = to_categorical(train.labels).shape[-1]
        print('beginning ' +loc )
        inputs = Input((52,8))
        x = Bidirectional(LSTM(128, dropout = drop, recurrent_dropout=rec_drop))(inputs)
        x = PReLU()(x)
        #outputs = Dense(150)(x)
        outputs = Dense(out_shape, activation='softmax')(x)
        lstm = Model(inputs, outputs)
        lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        lstm.fit(train, epochs=100,
                callbacks=[clr, EarlyStopping(patience=1000, monitor='val_loss')],
                validation_data=test, shuffle = False, workers=12, use_multiprocessing=True)
        results[loc] = lstm.evaluate(test)
        lstm.save("models/lstm_{}.h5".format(loc))
        plt.subplot(212)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.subplot(211)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0]*2, Size[1]*2)
        plt.savefig("training_{}.png".format(loc))
        print(results)


