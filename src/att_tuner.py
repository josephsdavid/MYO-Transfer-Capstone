import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
tf.autograph.set_verbosity(0)
import multiprocessing
import numpy as np
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from tensorflow.keras import backend as K
import callbacks as cb
import utils as u
import builders.recurrent as b
import kerastuner as kt
import losses as l
from layers import LayerNormalization
from optimizers import Ranger
from activations import Mish
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D, Flatten, MaxPooling1D, LSTM, Lambda, Permute, Reshape, Multiply, RepeatVector
'''
the batch size: 512 for now, may change to smaller later
scale learning rate with batch size
'''
batch=128


train = u.NinaMA("../data/ninaPro", ['a','b','c'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['a','b','c'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
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


def build_real_attention(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1], model_id=None):
    inputs = Input((n_time, 16))
    x = inputs
    x = Dense(128, activation=Mish())(x)
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


def build_model(hp):

    gamma=hp.Float('gamma', 2, 12, step=1)
    alpha=hp.Float('alpha', 2, 12, step=1)
    drop = hp.Float('dropout', 0.2, 0.5)

    model,a = build_real_attention(n_time, n_class, [256, 512, 1024], drop = [drop for _ in range(3)])
    loss = l.focal_loss( gamma=gamma, alpha=alpha)
    model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    return model


stopper = EarlyStopping(monitor = "val_loss", patience=20)
tuner = kt.Hyperband(build_model, objective = 'val_accuracy', max_epochs=55, hyperband_iterations = 3, directory = 'result', project_name = 'att_final')

tuner.search_space_summary()


import pdb; pdb.set_trace()  # XXX BREAKPOINT
tuner.search(train, validation_data = test,  callbacks = [cb.CosineAnnealingScheduler(T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5)], shuffle=False)

best = tuner.get_best_hyperparameters(1)[0]
print(best.values)
import json
with open("result/best_att_pars.json","w") as f:
	json.dump(best.values, f)

model = tuner.hypermodel.build(best)
u.save_model(model,"result/best_att")




