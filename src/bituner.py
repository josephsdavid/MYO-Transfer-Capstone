import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
import kerastuner as kt
from tensorflow.keras.models import load_model
from scipy.stats import gaussian_kde
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Embedding, Activation, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


batch = 300
train = u.NinaGenerator("../data/ninaPro", ['c'], [u.butter_highpass_filter],
                        [u.add_noise_snr], validation=False, by_subject = True, batch_size=batch,
                        scale = False, rectify=False)
test = u.NinaGenerator("../data/ninaPro", ['c'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = True, batch_size=batch,
                       scale = False, rectify = False)
n_classes=24

def build_model(hp):
    drop = hp.Float("dropout",0,0.5)
    rdrop = hp.Float("rdropout",0,0.5)
    inputs = Input((52,8))
    lstm_size = hp.Int('lstms', 32, 256, step=32)
    x, s1, s2, s3, s4 = Bidirectional(LSTM(lstm_size, dropout = drop, recurrent_dropout=rdrop, return_sequences=True, return_state=True, bias_initializer='ones'))(inputs)
    x = PReLU()(x)
    x = Bidirectional(LSTM(lstm_size, dropout = drop, recurrent_dropout=rdrop, return_sequences=False))(x, initial_state = [s1,s2,s3,s4])
    x = PReLU()(x)
    outputs=Dense(n_classes, activation='softmax')(x)

    mod = Model(inputs, outputs)
    opt=SGD(learning_rate=1e-3, momentum=0.9, nesterov=True, decay=1e-4)
    mod.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return mod

stopper= EarlyStopping(monitor='val_loss', patience=10)

tuner=kt.Hyperband(build_model, objective='val_accuracy', max_epochs=100, hyperband_iterations=3,
                   directory = 'result', project_name = 'blstm_tuning', distribution_strategy=tf.distribute.MirroredStrategy())

tuner.search(train, validation_data = test, callbacks=[stopper], shuffle = False)


best = tuner.get_best_hyperparameters(1)[0]
print(best.values)
import json
with open("result/best_bi_pars.json","w") as f:
	json.dump(best.values, f)
model = tuner.hypermodel.build(best)
model.save("result/best_bi_lstm.h5")
