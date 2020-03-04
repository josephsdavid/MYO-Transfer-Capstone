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




#import pdb; pdb.set_trace()  # XXX BREAKPOINT



train = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
val = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

test = u.TestGen(*val.test_data, shuffle=False, batch_size=batch)

def get_weight(arr, idx):
    return arr.labels[np.where(arr.labels.argmax(-1)==idx)].shape[0] / arr.labels[np.where(arr.labels.argmax(-1)!=idx)].shape[0]


weight_dict = {i:1/get_weight(train,i) for i in range(train[0][1].shape[-1])}

#factor=1.0/sum(weight_dict.values())
#for k in weight_dict:
#  weight_dict[k] = weight_dict[k]*factor

alpha = train.labels[np.where(train.labels.argmax(-1)==0)].shape[0] / train.labels[np.where(train.labels.argmax(-1)!=0)].shape[0]


#cyclic = cb.CyclicLR(step_size=2*len(train), mode='triangular2')

n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

import pdb; pdb.set_trace()  # XXX BREAKPOINT

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
    x = LayerNormalization()(x)
    x, a = Attention(return_attention=True)(x)
    for d, dr in zip(dense, drop):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, a)




def build_conv_rnn(n_time, n_classes, filters = [20,64,64,64], kernels=[7,5,5,5], lstm=[52,52]):
    '''
    build_conv_rnn
    ----------------
    builds a conv_rnn
    args:
        n_time: timesteps,
        n_classesL num classes
        filters: list of filters to use
        kernels=list of kernels to use
        lstm: list of lstm sizes
    outputs:
        not compiled model
    '''
    inputs = Input((n_time, 16))
    x = inputs
    for f, k in zip(filters, kernels):
        x = (Conv1D(filters=f, kernel_size=k, activation=Mish()))(x)
    x = (Dropout(0))(x)
    x = (MaxPooling1D())(x)
    #x = TimeDistributed(Flatten())(x)
    x, h, b = LSTM(lstm[0], activation=Mish(), return_sequences=True, return_state=True)(x)
    a1, s1 = Attention(return_attention=True)(x)
    x, h , b= LSTM(lstm[0], activation=Mish(), return_sequences=True, return_state=True)(x, initial_state=[h,b])
    a2, s2 = Attention(return_attention=True)(x)
    #for l in range(len(lstm)):
    #    seq = True if (l != len(lstm)-1) else False
    #    x = GRU(lstm[l], dropout=0.2, return_sequences=True)(x)
    x = Add()([a1,a2])
    a = Add()([s1,s2])
    x = Dense(100, activation=Mish())(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs), Model(inputs, a)









#model, att = build_conv_rnn(n_time, n_class)
#model = build_att_gru(n_time, n_class, nodes=128)

def attention_3d(inputs, n_time):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='temporalize')(inputs)
    #a = Reshape((input_dim, n_time))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(n_time, activation='softmax',  name='attention_probs')(a)
    a = Lambda(lambda x: K.mean(x, axis=1))(a)
    a = RepeatVector(input_dim)(a)
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



model, attn = build_real_attention(n_time, n_class, [256, 512, 1024], drop = [ 0.5, 0.5, 0.5])

tf.keras.utils.plot_model(model, to_file="att_flat.png", show_shapes=True, expand_nested=True)

cosine = cb.CosineAnnealingScheduler(T_max=50, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5)
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
print(model.summary())

#model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model.fit(train, epochs=55, validation_data=val, shuffle=False,
               callbacks=[ModelCheckpoint("att_class_weights.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True),
                          cosine], use_multiprocessing=True, workers=50,
               max_queue_size=1000, class_weight = weight_dict
               )

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
plt.savefig("att_weighted_uvar.png")
import pdb; pdb.set_trace()  # XXX BREAKPOINT


# stop here
h2 = model.fit(train, epochs=100, validation_data=val, shuffle=False,
               callbacks=[ModelCheckpoint("att_forward_small.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True)
                          ], use_multiprocessing=True, workers=50, initial_epoch=55,
               max_queue_size=1000
               )

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
plt.savefig("att_real_univariate.png")
