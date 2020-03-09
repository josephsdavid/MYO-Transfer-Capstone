import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D, Flatten, MaxPooling1D, LSTM, Lambda, Permute, Reshape, Multiply, RepeatVector
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
import tensorflow.keras.backend as K
batch=128

train = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, n=15, window_size=52, super_augment=False, imu=True)
val = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False, shuffle=False, imu=True)
test = u.TestGen(*val.test_data, shuffle=False, batch_size=batch)

import pdb; pdb.set_trace()  # XXX BREAKPOINT

n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]


def attention_3d(inputs, n_time):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='temporalize')(inputs)
    #a = Reshape((input_dim, n_time))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(n_time, activation='softmax',  name='attention_probs')(a)
  #  a = Lambda(lambda x: K.mean(x, axis=1))(a)
  #  a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat, a_probs

def build_conv_attention(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1], model_id=None):
    inputs = Input((n_time, 19))
    x = inputs
    #x=Dense(250, activation=Mish())(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_3d(x, n_time)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, Dense(16)(a))


model, attn = build_conv_attention(38, 53, [500,500,2000], drop = [0.36 for _ in range(3)])
tf.keras.utils.plot_model(model, to_file="att_flat.png", show_shapes=True, expand_nested=True)
cosine = cb.CosineAnnealingScheduler(T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5)
loss = l.focal_loss( gamma=3., alpha=6.)
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
print(model.summary())
import pdb; pdb.set_trace()  # XXX BREAKPOINT
h2 = model.fit(train, epochs=55, validation_data=val, shuffle=False,
               callbacks=[ModelCheckpoint("att_conv128-3-imu.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True),
                          cosine], use_multiprocessing=True, workers=50,
               max_queue_size=1000, steps_per_epoch = len(train))

