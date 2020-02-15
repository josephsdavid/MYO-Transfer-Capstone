import utils as u
import numpy as np
from optimizers import Ranger, Yogi, Lookahead
from activations import Mish
from tensorflow.keras.layers import Dense, Input, SimpleRNN, PReLU, Add, BatchNormalization, RepeatVector, Flatten, TimeDistributed
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
        n=5):
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

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:]
        if self.augmentors is not None:
            for f in self.augmentors:
                for i in range(out.shape[0]):
                    out[i,:,:]=f(out[i,:,:])
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)
        out = np.moveaxis(ma_batch(out, self.n), -1, 0)
        return out,  self.labels[indexes]


''' begin of analysis in earnest '''



import pdb; pdb.set_trace()  # XXX BREAKPOINT

train = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=1)
test = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=1)

n_time = train[0][0].shape[1]
print("n_timesteps{}".format(n_time))


neg = Constant(value=-1)


inputs = Input((n_time, 8))
x = (Dense(128))(inputs)
x, h = SimpleRNN(20, activation=PReLU(), name='simple1',  return_state=True, return_sequences=True)(x)
rnns = []
rnns.append(x)
for i in range(3):
    x, h = SimpleRNN(20, activation=PReLU(), return_state=True, return_sequences=True)(x, initial_state=[h])
    rnns.append(x)
out = Add()(rnns)
out = Flatten()(out)
out = Dense(60, activation=PReLU())(out)
outputs = Dense(18, activation="softmax")(out)
model = Model(inputs, outputs)
model.summary()
model.compile(Lookahead(RMSprop()), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
h1 = model.fit(train, epochs=500, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("rnn.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12)

plt.subplot(212)
plt.plot(h1.history['accuracy'])
plt.plot(h1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(211)
plt.plot(h1.history['loss'])
plt.plot(h1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2)
plt.savefig("simple_lstm_training.png")

train = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15)
test = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15)

n_time = train[0][0].shape[1]
print("n_timesteps{}".format(n_time))


neg = Constant(value=-1)


inputs = Input((n_time, 8))
x = (Dense(128))(inputs)
x, h = SimpleRNN(20, activation=PReLU(), name='simple1',  return_state=True, return_sequences=True)(x)
rnns = []
rnns.append(x)
for i in range(3):
    x, h = SimpleRNN(20, activation=PReLU(), return_state=True, return_sequences=True)(x, initial_state=[h])
    rnns.append(x)
out = Add()(rnns)
out = Flatten()(out)
out = Dense(60, activation=PReLU())(out)
outputs = Dense(18, activation="softmax")(out)
model = Model(inputs, outputs)
model.summary()
model.compile(Lookahead(RMSprop()), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
h2 = model.fit(train, epochs=500, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("rnn2.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12)


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
