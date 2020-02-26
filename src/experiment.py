import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
batch=512




#import pdb; pdb.set_trace()  # XXX BREAKPOINT



train = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
val = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

test = u.TestGen(*val.test_data, shuffle=False, batch_size=batch)


#cyclic = cb.CyclicLR(step_size=2*len(train), mode='triangular2')

n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

loss = l.focal_loss( gamma=4.)

print("n_timesteps: {}".format(n_time))

neg = Constant(value=-1)

def gru(inn, nodes=40, dropout=0,**kwargs):
    return GRU(nodes, activation=PReLU(),  return_state=True, return_sequences=True, reset_after=True, recurrent_activation='sigmoid', recurrent_dropout=dropout)(inn, **kwargs)


def block(inn, nodes=40, dropout = 0,**kwargs):
    val, state = gru(inn, nodes, dropout, **kwargs)
    val2 = Attention()(Dropout(0)(val))
    return val, Dropout(0)(state), val2


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
    inputs = Input(shape = (n_time, 16))
    d = 0
    x = Dense(128)(inputs)
    #x = Conv1D(28, kernel_size=7, strides=1, padding='valid', activation=Mish())(inputs)
    #x = Conv1D(28, kernel_size=5, strides=1, padding='valid', activation=Mish())(x)
    #x = Conv1D(28, kernel_size=2, strides=1, padding='valid', activation=Mish())(x)
    #x = Conv1D(28, kernel_size=3, strides=2, padding='valid', activation=Mish())(x)
    x = Dropout(0)(x)
    x, h, a = block(x, nodes, dropout=d)
    attention=[a]
    for _ in range(blocks-1):
        x = Dropout(0)(x)
        x, h, a = block(x, nodes, dropout=d,initial_state=h)
        attention.append(a)
    out = Add()(attention)
    out = Dropout(0)(out)
    outputs = Dense(n_classes, activation="softmax")(out)
    model = Model(inputs, outputs)
    model.compile(optimizer(**optim_args), loss=loss,  metrics=['accuracy'])
    print(model.summary())
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model



#model = br.build_att_gru(n_time, n_class, nodes=40,loss = loss)
#model = bc.build_cnn(n_time, n_class, filters=[20, 64, 64, 64, 64, 64])


def build_simple_att(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1], model_id=None):
    '''
    build_simple_att
    ------------------
    args:
        n_time
        n_class
        dense = [50,50,50], list of dense node sizes
        drop = [0.1, 0.1, 0.1] list of dropout
    '''
    inputs = Input((n_time, 16))
    x = inputs
    x = Dense(128, activation=Mish())(x)
    x = Attention()(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model

#model = build_simple_att(n_time, n_class, [256, 512, 1024])
model = build_att_gru(n_time, n_class, nodes=128)
tf.keras.utils.plot_model(model, to_file="att_gru_drop.png", show_shapes=True, expand_nested=True)
cosine = cb.CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=5e-5, verbose=0, epoch_start=10)
model.compile(Ranger(), loss=loss, metrics=['accuracy'])
print(model.summary())
#model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
stopper = EarlyStopping(monitor = "val_loss", patience=20)
h2 = model.fit(train, epochs=100, validation_data=val, shuffle=False,
               callbacks=[ModelCheckpoint("att_forward_small.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True), cosine, stopper], use_multiprocessing=True, workers=12
               )

import pdb; pdb.set_trace()  # XXX BREAKPOINT
model.evaluate(test)

import matplotlib.pyplot as plt
plt.subplot(212)
plt.plot(h2.history['accuracy'])
plt.plot(h2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
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
plt.savefig("agru.png")
