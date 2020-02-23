import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout
#from builders.recurrent import build_att_gru_norm, build_att_gru
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
batch=512



import pdb; pdb.set_trace()  # XXX BREAKPOINT



train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

#cyclic = cb.CyclicLR(step_size=2*len(train), mode='triangular2')

n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

loss = l.focal_loss( gamma=4., alpha=8.)

print("n_timesteps: {}".format(n_time))

neg = Constant(value=-1)

def gru(inn, nodes=40, **kwargs):
    return GRU(nodes, activation='tanh',  return_state=True, return_sequences=True, reset_after=True, recurrent_activation='sigmoid', recurrent_dropout=0.1, dropout=0.1)(inn, **kwargs)


def block(inn, nodes=40,**kwargs):
    val, state = gru(inn, nodes,**kwargs)
    val2 = Attention()(val)
    return val, state, val2


def build_att_gru(n_time, n_classes, nodes=40, blocks=3,
                  loss='categorical_crossentropy', optimizer=Ranger,model_id=None, **optim_args):
    '''
    build_att_gru
    --------------
    args:
        n_time
        n_out
        nodes=40
        blocks=3, represents number of att_gru blocks in the model
        loss='categorical_crossentropy'
        optimizer=Ranger (no parentheses)
        model_id = model id
        args for optimizer
    Notes:
        returns compiled model.
        requires by default one hot encoded Y data
    '''
    inputs = Input((n_time, 8))
    x = Dense(128, activation=Mish())(inputs)
    x, h, a = block(x, nodes)
    attention=[a]
    for _ in range(blocks-1):
        x, h, a = block(x, nodes, initial_state=h)
        attention.append(a)
    out = Add()(attention)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    print(model.summary())
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model



model=build_att_gru(n_time, n_class, loss = loss)
tf.keras.utils.plot_model(model, to_file="attn.png", show_shapes=True, expand_nested=True)

#model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model.fit(train, epochs=100, validation_data=test, shuffle=False,
               callbacks=[ModelCheckpoint("gru2.h5", monitor="val_loss", keep_best_only=True)], use_multiprocessing=True, workers=12
               )


import matplotlib.pyplot as plt
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
plt.savefig("lstm_untuned.png")
