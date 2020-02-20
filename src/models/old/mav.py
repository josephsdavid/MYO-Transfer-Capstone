import utils as u
import multiprocessing
import numpy as np
from optimizers import Ranger, Yogi, Lookahead
from layers import Attention
from activations import Mish
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GRU, PReLU, Add, BatchNormalization, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
batch=512

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def ma(window, n):
    return np.vstack([moving_average(window[:,i], n) for i in range(window.shape[-1])]).T

def ma_batch(batch, n):
        return np.dstack([ma(batch[i,:,:], n) for i in range(batch.shape[0])])


def scale(arr3d):
    for i in range(arr3d.shape[0]):
        arr3d[i,:,:] /= np.abs(arr3d[i,:,:]).max(axis=0)
    return arr3d


class NinaMA(u.NinaGenerator):
    def __init__(self,
        path: str,
        excercises: list,
        process_fns: list,
        augment_fns: list,
        scale=False,
        rectify=False,
        step=5,
        window_size=52,
        batch_size=400, shuffle=True,
        validation=False,
        by_subject=False,
        sample_0=True,
        n=5, super_augment = False):
        super().__init__(
        path,
        excercises,
        process_fns,
        augment_fns,
        scale,
        rectify,
        step,
        window_size,
        batch_size, shuffle,
        validation,
        by_subject,
        sample_0
        )
        self.n = n
        self.super_augment = super_augment
        if super_augment:
            if self.augmentors is not None:
                o = [self.emg]
                ls = [self.labels]
                rs = [self.rep]
                ss = [self.subject]
                for f in self.augmentors:
                    inn = [self.emg[i,:,:] for i in range(self.emg.shape[0]) if self.labels[i] != 0]
                    ls.append(self.labels[np.where(self.labels != 0)])
                    rs.append(self.rep[np.where(self.labels != 0)])
                    ss.append(self.subject[np.where(self.labels != 0)])
                    with multiprocessing.Pool(None) as p:
                        res = p.map(f, inn)
                    o.append(np.moveaxis(np.dstack(res), -1, 0))
                self.emg = np.concatenate(o, axis=0)
                self.labels = np.concatenate( ls, axis=0)
                self.rep = np.concatenate( rs, axis=0)
                self.subject = np.concatenate( ss, axis=0)
                print(self.labels.shape)
                print(self.rep.shape)
                print(self.subject.shape)
                print(self.emg.shape)
                print("okokok")
                self.on_epoch_end()



    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:].copy()
        if not self.super_augment:
            if self.augmentors is not None:
                for f in self.augmentors:
                    for i in range(out.shape[0]):
                        if self.labels[i] == 0:
                            out[i,:,:] = out[i,:,:]
                        else:
                            out[i,:,:]=f(out[i,:,:])
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)
        return np.moveaxis(ma_batch(out, self.n), -1, 0),  self.labels[indexes]


''' begin of analysis in earnest '''



import pdb; pdb.set_trace()  # XXX BREAKPOINT

#train = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
#                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
#                        scale = False, rectify=True, sample_0=False, step=5, n=1)
#test = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
#                       None, validation=True, by_subject = False, batch_size=batch,
#                       scale = False, rectify =True, sample_0=False, step=5, n=1)
#
#n_time = train[0][0].shape[1]
#print("n_timesteps{}".format(n_time))
#
#
#neg = Constant(value=-1)
#
#
#inputs = Input((n_time, 8))
#x = (Dense(128))(inputs)
#x, h = GRU(20, activation=PReLU(), name='simple1',  return_state=True, return_sequences=True)(x)
#rnns = []
#rnns.append(x)
#for i in range(3):
#    x, h = GRU(20, activation=PReLU(), return_state=True, return_sequences=True)(x, initial_state=[h])
#    rnns.append(x)
#out = Add()(rnns)
#out = Flatten()(out)
#out = Dense(60, activation=PReLU())(out)
#outputs = Dense(18, activation="softmax")(out)
#model = Model(inputs, outputs)
#model.summary()
#model.compile(Lookahead(RMSprop()), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#h1 = model.fit(train, epochs=500, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("rnn.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12)
#
#plt.subplot(212)
#plt.plot(h1.history['accuracy'])
#plt.plot(h1.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
## summarize history for loss
#plt.subplot(211)
#plt.plot(h1.history['loss'])
#plt.plot(h1.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#F = plt.gcf()
#Size = F.get_size_inches()
#F.set_size_inches(Size[0]*2, Size[1]*2)
#plt.savefig("simple_lstm_training.png")

train = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, super_augment=False)
test = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15)


n_time = train[0][0].shape[1]
print("n_timesteps{}".format(n_time))
#from collections import Counter
#import pdb; pdb.set_trace()  # XXX BREAKPOINT


neg = Constant(value=-1)


inputs = Input((n_time, 8))
x = (Dense(128))(inputs)
x, h = GRU(40, activation=PReLU(), name='res_a',  return_state=True, return_sequences=True)(x)
l = []
l.append(x)
x1, h1 = GRU(40, activation=PReLU(), name='res_b',  return_state=True, return_sequences=True)(x, initial_state=[h])
l.append(x1)
for i in range(3):
    xi, h1 = GRU(40, activation=PReLU(), name='res_{}'.format(i),  return_state=True, return_sequences=True)(Add()([l[-1], l[-2]]))
    l.append(xi)
xi = Add()([l[-1],l[-2]])
#x, h = GRU(20, activation=PReLU(), name='simple1',  return_state=True, return_sequences=True)(x)
#x1, h1 = GRU(20, activation=PReLU(), name='simple2',  return_state=True, return_sequences=True)(x, initial_state=[h])
#x2, h2 = GRU(20, activation=PReLU(), name='simple3',  return_state=True, return_sequences=True)(Add()([x, x1]), initial_state=[h1])
#x3, h3 = GRU(20, activation=PReLU(), name='simple4',  return_state=True, return_sequences=True)(Add()([x1, x2]), initial_state=[h2])
#x4, h4 = GRU(20, activation=PReLU(), name='simple5',  return_state=True, return_sequences=True)(Add()([x2, x3]), initial_state=[h3])
#x4 = Add()([x4, x3])
#rnns = []
#rnns.append(x)
#for i in range(3):
#    if i == 0:
#        x, h = GRU(20, activation=PReLU(), return_state=True, return_sequences=True)(x, initial_state=[h])
#    else:
#        x, h = GRU(20, activation=PReLU, return_state=True, return_sequences=True)(Add()(rnns), initial_state=[h])
#    rnns.append(x)
out = Attention()(xi)
#out = Dense(60, activation=PReLU())(out)
outputs = Dense(18, activation="softmax")(out)
model = Model(inputs, outputs)
model.summary()

tf.keras.utils.plot_model(model, to_file="attn.png")
model.compile(Lookahead(RMSprop()), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

h2 = model.fit(train, epochs=500, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("gru2.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=60, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12, steps_per_epoch=len(train)//5)


plt.subplot(212)
plt.plot(h2.history['accuracy'])
plt.plot(h2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(211)
plt.plot(h2.history['loss'])
plt.plot(h2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2)
plt.savefig("simple_lstm_training.png")
