import utils as u
import multiprocessing
import numpy as np
from optimizers import Ranger, Yogi, Lookahead
import callbacks as cb
from layers import Attention
from activations import Mish, sparsemax, SparsemaxLoss
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GRU, PReLU, Add, BatchNormalization, RepeatVector, Flatten, TimeDistributed, Subtract, Multiply, Average, Maximum
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from builders import build_att_gru
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
        self.labels = to_categorical(self.labels)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:].copy()
        if not self.super_augment:
            if self.augmentors is not None:
                for f in self.augmentors:
                    for i in range(out.shape[0]):
                        if self.labels[i,0] == 1:
                            out[i,:,:] = out[i,:,:]
                        else:
                            out[i,:,:]=f(out[i,:,:])
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)
        return np.moveaxis(ma_batch(out, self.n), -1, 0),  self.labels[indexes,:]


''' begin of analysis in earnest '''





import pdb; pdb.set_trace()  # XXX BREAKPOINT


train = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


n_time = train[0][0].shape[1]

n_class =train[0][1].shape[-1]

print("n_timesteps{}".format(n_time))

neg = Constant(value=-1)

# define layer wrappers for ease:
'''
def gru(inn, **kwargs):
    # this is the only way it works!
    return GRU(40, activation=Mish(),  return_state=True, return_sequences=True)(inn, **kwargs)
def block(inn, **kwargs):
    val, state = gru(inn, **kwargs)
    val2 = Attention()(val)
    return val, state, val2

inputs = Input((n_time, 8))
x = Dense(128)(inputs)
x1, h, a1 = block(x)
x2, h, a2 = block(x1, initial_state=h)
x3, h, a3 = block(x2, initial_state=h)
out = Add()([a1,a2,a3])
#out = Dense(60, activation=PReLU())(out)
outputs = Dense(18, activation="softmax")(out)
model = Model(inputs, outputs)
model.summary()
'''

model=build_att_gru(n_time, n_class, learning_rate=0.01)
tf.keras.utils.plot_model(model, to_file="attn.png", show_shapes=True, expand_nested=True)
model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
class_weights = {i:1/(n_class) if i==0 else 1 for i in range(1, n_class+1)}
h2 = model.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("gru2.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12, class_weight=class_weights)


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
