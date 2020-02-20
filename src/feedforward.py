import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
tf.autograph.set_verbosity(0)
import multiprocessing
import numpy as np
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
import callbacks as cb
import utils as u
from optimizers import Ranger
from activations import Mish
import builders
batch=512


train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

model = builders.build_simple_att(n_time, n_class, drop=[0,0,0], dense=[50,50,50], activation=Mish())
model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


json_file = model.to_json()
id="simple_att"
with open(f"{id}.json", "w") as f:
    f.write(json_file)


class_weights = {i:1/(n_class) if i==0 else 1 for i in range(1, n_class+1)}
h2 = model.fit(train, epochs=100, validation_data=test, shuffle=False,
               callbacks=[ModelCheckpoint(f"{id}.h5", monitor="val_loss", keep_best_only=True, save_weights_only=True),
                          ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True,
               workers=12, class_weight=class_weights)
