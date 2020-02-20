import utils as u
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from lsmtfcn import build_fcnn_lstm
import tensorflow as tf
import callbacks as cb
import numpy as np
import pdb; pdb.set_trace()  # XXX BREAKPOINT
batch=1000
#train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
#                        [u.add_noise_snr], validation=False, by_subject = True, batch_size=batch,
#                        scale = False, rectify=True, sample_0=True)
#test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
#                       None, validation=True, by_subject = True, batch_size=batch,
#                       scale = False, rectify = True, sample_0=False)
abc = ['b','a','c']
subject=[False, True]
clr=cb.OneCycleLR(
    max_lr=0.5,
    end_percentage=0.1,
    scale_percentage=None,
    maximum_momentum=0.95,
    minimum_momentum=0.85,
    verbose=True)
results = {}
hl = []
strategy = tf.distribute.MirroredStrategy()

def holder(x,y):
    return x, y

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch, scale=True)
test_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = batch, scale=True)
ncl = 7
n_smpl = 7
lstm = build_fcnn_lstm(ncl)
lstm.compile(SGD(),'sparse_categorical_crossentropy', metrics=['accuracy'])
lstm.fit(train_set, steps_per_epoch = len(train_set)//4, epochs=25, validation_data=test_set, shuffle=False, callbacks = [clr, ModelCheckpoint("src.h5", monitor="val_loss", keep_best_only=True)])


for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=0.5,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch,
                                scale = True, rectify=False, sample_0=False)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch,
                               scale = True, rectify = False, sample_0=False)
        nb_classes = np.unique(train.labels).shape[0]
        sub = "subject" if s else "rep"
        lstm = load_model("src.h5")
        lstm.trainable=True
        hidden = Dense(120, name = "harold")(lstm.layers[-2].output)
        out=Dense(nb_classes, activation='softmax', name = 'class')(hidden)
        lstm = Model(lstm.input, out)
        lstm.compile(SGD(),'sparse_categorical_crossentropy', metrics=['accuracy'])
        print("beginning transfer_fine_{}_{}".format(abc[i], sub))
        hl.append(lstm.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks = [clr, ModelCheckpoint("fcnlstm_transfer_fine_{}_{}.h5".format(sub, abc[i]), monitor="val_loss", keep_best_only=True)]))


for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=0.5,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch,
                                scale = True, rectify=False, sample_0=False)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch,
                               scale = True, rectify = False, sample_0=False)
        nb_classes = np.unique(train.labels).shape[0]
        sub = "subject" if s else "rep"
        lstm = build_fcnn_lstm(nb_classes)
        lstm.compile(SGD(),'sparse_categorical_crossentropy', metrics=['accuracy'])
        print("beginning notransfer_{}_{}".format(abc[i], sub))
        hl.append(lstm.fit(train, epochs=100, validation_data=test, shuffle=False,
                           callbacks = [clr, ModelCheckpoint("fcnlstm_notransfer_{}_{}.h5".format(sub, abc[i]), monitor="val_loss", keep_best_only=True)]))



for i in range(len(abc)):
    for s in subject:
        clr=cb.OneCycleLR(
            max_lr=0.5,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        train = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = s, batch_size=batch,
                                scale = True, rectify=False, sample_0=False)
        test = u.NinaGenerator("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch,
                               scale = True, rectify = False, sample_0=False)
        nb_classes = np.unique(train.labels).shape[0]
        sub = "subject" if s else "rep"
        lstm = load_model("src.h5")
        lstm.trainable=False
        hidden = Dense(120, name = 'big')(lstm.layers[-2].output)
        out=Dense(nb_classes, activation='softmax', name = 'class')(hidden)
        lstm = Model(lstm.input, out)
        lstm.compile(SGD(),'sparse_categorical_crossentropy', metrics=['accuracy'])
        print("beginning transfer_cold_{}_{}".format(abc[i], sub))
        hl.append(lstm.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks = [clr, ModelCheckpoint("fcnlstm_cold_transfer_{}_{}.h5".format(sub, abc[i]), monitor="val_loss", keep_best_only=True)]))
