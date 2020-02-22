import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input,Add, Concatenate, Dense, GRU, TimeDistributed
from layers import Attention, LayerNormalization
from activations import Mish
from optimizers import Ranger
import utils as u

batch=512

train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


outs = [[] for i in range(4)]
for tr in train:
    outs[0].append(tr[0])
    outs[1].append(tr[1])
# augmentation is joyous
for tr, te in zip(train, test):
    outs[0].append(tr[0])
    outs[1].append(tr[1])
    outs[2].append(te[0])
    outs[3].append(te[1])

x_tr, y_tr, x_val, y_val = (np.concatenate(o, 0) for o in outs)

tr_dict = {'class':y_tr, 'encdec':x_tr}
tr_dict = {'class':y_val, 'encdec':x_val}

n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

latent_dim=2
layers = [50, 75, 100]

inputs = Input((n_time, 8))
x = inputs
a = []
for l in layers:
    x = GRU(l, activation=Mish())(x)
    a += [Attention()(x)]
encoded = Dense(latent_dim)(x)
encoded_a = Attention()(encoded)
a.append(encoded_a)
x = encoded
for l in layers[::-1]:
    x = GRU(l, activation=Mish())(x)
    a += [Attention()(x)]
att = Concatenate()(a)
#att = Dense(100)(att)
ae_out = TimeDistributed(Dense(8, activation=Mish()), name='encdec')(x)
out = Dense(n_class, activation='softmax', name='class')(att)
encoder = Model(inputs, encoded_a)
ae = Model(inputs, [])
ae.compile(Ranger(), loss='categorical_crossentropy',  metrics=['accuracy'])

tf.keras.utils.plot_model(ae, to_file="encoder.png", show_shapes=True, expand_nested=True)

class_weights = {i:1/(n_class) if i==0 else 1 for i in range(1, n_class+1)}
h2 = ae.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("ae.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12, class_weight=class_weights)
