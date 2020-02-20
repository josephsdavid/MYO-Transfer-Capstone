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
import builders.attention as b
import kerastuner as kt
batch=512


train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]


def build_model(hp):
    lr = hp.Float("learning_rate", 1e-5, 10, sampling='log', default=1e-3)
    wd = hp.Float("decay", 0., 0.5, step=0.05, default=0.)
    sp = hp.Int("sync", 2, 10, step=1, default=6)
    wp = hp.Float("warm", 0.1, 0.3, step=0.05, default=0.1)
    slow_step = hp.Float("slow_step_size", 0.2, 0.8, step=0.1, default=0.5)
    ams=hp.Boolean('ams')
    model = b.build_att_gru(n_time, n_class,
                          learning_rate=lr, weight_decay=wd, sync_period=sp,
                          warmup_proportion=wp, slow_step_size=slow_step,
                          amsgrad=ams)
    return model


stopper = EarlyStopping(monitor = "val_loss", patience=20)
tuner = kt.Hyperband(build_model, objective = 'val_accuracy', max_epochs=100, hyperband_iterations = 3, directory = 'result', project_name = 'att_tune')

tuner.search_space_summary()

import pdb; pdb.set_trace()  # XXX BREAKPOINT
tuner.search(train, validation_data = test,  callbacks = [stopper], shuffle=False)

best = tuner.get_best_hyperparameters(1)[0]
print(best.values)
import json
with open("result/best_att_pars.json","w") as f:
	json.dump(best.values, f)

model = tuner.hypermodel.build(best)
u.save_model(model,"restult/best_att")




