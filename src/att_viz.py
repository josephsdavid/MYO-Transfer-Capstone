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
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D, Flatten, MaxPooling1D, LSTM
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
batch=128



def build_montecarlo_att(n_time, n_class, dense = [50,50,50], drop=[0.1, 0.1, 0.1], model_id=None):
    '''
    build_montecarlo_att
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
    x, a = Attention(return_attention=True)(x)
    for d, dr in zip(dense, drop):
        # note the training arg, using dropout at prediction time is over cooly
        # referred to as monte carlo output
        x = Dropout(dr)(x, training=True)
        x = Dense(d, activation=Mish())(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, a)


val = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.TestGen(*val.test_data, shuffle=False, batch_size=batch, zeros=False)
del val
X = test.X
Y = test.y.argmax(-1)
#X=np.concatenate(test.X, axis=0)
#X = np.moveaxis(u.window_roll(X, 1, 38), -1,1)

n_time = test[0][0].shape[1]
n_class =test[0][1].shape[-1]
import gc; gc.collect()



model, scorer = build_montecarlo_att(n_time, n_class, [256, 512, 1024], drop = [0.1, 0.1, 0.1], model_id="att_forward_small")
loss = l.focal_loss( gamma=4.)
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=['accuracy'])



# uncertainty = std(predictions), see "dropout as a bayesian estimator paper"
def predict(model, x, n_classes=53, n_iter=100):
    result = np.zeros((n_iter,) + (x.shape[0], n_classes))
    for i in range(n_iter):
        print(i)
        result[i,:,:] = model.predict(x)
    prediction = result.mean(axis=0)
    uncertain = result.std(axis=0)
    return prediction, uncertain


preds, uncs = predict(model, X, n_iter=100)


# without the dropout
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
    x, a = Attention(return_attention=True)(x)
    for d, dr in zip(dense, drop):
        # note the training arf
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
    outputs = Dense(n_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    if model_id is not None:
        model.load_weights(f"{model_id}.h5")
    return model, Model(inputs, a)

#model, scorer = build_simple_att(n_time, n_class, [256, 512, 1024], drop = [0.5, 0.5, 0.5], model_id="att_forward_small")
#loss = l.focal_loss( gamma=4.)
#model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=['accuracy'])



import matplotlib
matplotlib.use('agg')
# bmh is the best
plt.style.use('bmh')
for i in range(X.shape[0]):
    # i am the worst
    print(f'{i+1}')
    # y axis, we need to have a batch shape/3d array, so it is 1,n_timesteps,
    # n_features
    y = X[i][np.newaxis]
    # we need a 2d array of x variables, one for each dimension of y variables
    x = np.vstack([np.arange(y.shape[1]) for _ in range(y.shape[2])]).T
    # get attention scores
    att_scores=scorer.predict(y)
    #
    t = np.vstack([att_scores[0] for _ in range(y.shape[2])]).T
    t -=t.min()
    t /= t.ptp()
    plt.scatter(x,X[i], c= t, cmap='inferno')
    plt.plot(x,X[i], 'b--', linewidth=0.2)
    plt.ylim(-10,80)
    best_pred = preds[i].argmax()
    txtstr=f'class: {Y[i]}, prediction: {best_pred} ({preds[i, best_pred]:.02f}), uncertainty: {uncs[i,best_pred]:.02f}'
    props = dict(boxstyle='round', facecolor='none', alpha=1)
    plt.text(-0.5, 75, txtstr,  fontsize=9, verticalalignment='top', bbox=props)
    plt.savefig('att_plot/{:07d}.png'.format(i+1))
    plt.clf()

