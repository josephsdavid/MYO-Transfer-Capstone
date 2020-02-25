import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import losses as l
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Add, Input, Dense, GRU, PReLU, Dropout, TimeDistributed, Conv1D, Activation, BatchNormalization, LeakyReLU
import builders.recurrent as br
import builders.attentional as ba
import builders.conv as bc
from activations import Mish
from optimizers import Ranger
from layers import Attention, LayerNormalization
batch=512
disable_eager_execution()


def build_models(n_time, n_class, nodes = 128):
    inputs = Input(shape=(n_time, 16))
    x1 = Attention()(inputs)
    x1 = Dense(nodes, activation='linear')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(LeakyReLU())(x1)

    source_class = Dense(n_class, activation='softmax', name = 'source')(x1)

    x = x1
    layers = [500, 500, 1000]
    for i in range(len(layers)):
        # build domain classifier
        x = Dense(layers[i],'linear', name = f"do{i}" )(x)
        x = BatchNormalization(name=f"do_bn{i}")(x)
        x = Activation(LeakyReLU(), name = f"do_act{i}")(x)

    domain_class = Dense(2, activation='softmax', name = "do")(x)

    combo = Model(inputs, outputs = [source_class, domain_class])
    combo.compile(optimizer=Ranger(),
                  loss = {
                      'source':l.focal_loss(gamma=4.),
                      'do':'categorical_crossentropy'
                  },
                  metrics=['accuracy'])

    source = Model(inputs=inputs, outputs=[source_class])
    source.compile(optimizer=Ranger(),
              loss={'source': l.focal_loss(gamma=4.)}, metrics=['accuracy'], )


    domain = Model(inputs=inputs, outputs=[domain_class])
    domain.compile(optimizer=Ranger(),
                  loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

    embeddings = Model(inputs=inputs, outputs=[x1])
    embeddings.compile(optimizer=Ranger(),loss = 'categorical_crossentropy', metrics=['accuracy'])

    return combo, source, domain, embeddings



'''now to fix up our train stuff'''



train_set = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = True, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
val_set = u.NinaMA("../data/ninaPro", ['a','b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = True, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


def train(training, validation, dann=True, n_epoch = 400):
    n_time = training[0][0].shape[1]
    n_class = training[0][1].shape[-1]
    model, source_model, domain_model, embeddings_model = build_models(n_time, n_class)
    print(model.summary())

    batch_size = validation.batch_size
    y_class_dummy = np.ones((len(validation), 2))
    y_adversarial = to_categorical(np.array(([1]*batch_size + [0]*batch_size)))

    sample_weights_class = np.array(([1] * (batch_size//2) + [0] * (batch_size//2)))
    sample_weights_adversarial = np.ones(((batch_size//2) * 2,))
    sample_weights = ([sample_weights_class, sample_weights_adversarial])




    for e in range(n_epoch):
        print(f"epoch: {e+1}")
        i = 1
        for source, target in zip(training, validation):
            print(i)
            i+=1
            y_adversarial2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

            X_adv = np.concatenate([source[0], target[0]])
            y_class = np.concatenate([source[1], np.zeros_like(source[1])])

            adv_weights = []
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    adv_weights.append(layer.get_weights())

            if dann:
                # try sample_weighting later on
                stats = model.train_on_batch(X_adv, [y_class, y_adversarial])

                k=0
                for layer in model.layers:
                    if (layer.name.startswith("do")):
                        layer.set_weights(adv_weights[k])
                        k+=1

                class_weights = []

                for layer in model.layers:
                    if (not layer.name.startswith("do")):
                        class_weights.append(layer.get_weights())

                stats2 = domain_model.train_on_batch(X_adv, [y_adversarial2])

                k=0
                for layer in model.layers:
                    if (not layer.name.startswith("do")):
                        layer.set_weights(class_weights[k])
                        k+=1
            else:
                source_model.train_on_batch(source[0], source[1])
        if e-1 % 4 == 0 and e != 0:
            print("train_score")
            source_model.evaluate(training)
            print("val_score")
            source_model.evaluate(validation)
        training.on_epoch_end()
        validation.on_epoch_end()
    return embeddings_model, source_model, domain_model

emb, so, do = train(train_set, val_set)

import pdb; pdb.set_trace()  # XXX BREAKPOINT

