from tensorflow.keras.layers import Dense, Input, GRU, PReLU
import utils as u
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

import pdb; pdb.set_trace()  # XXX BREAKPOINT

batch=100
train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_snr], validation=False, by_subject = False, batch_size=batch,
                        scale = True, rectify=False, sample_0=True, step=5)
test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = True, rectify = False, sample_0=False, step=5)


inputs = Input((52, 8))
x = GRU(200, activation=PReLU())(inputs)
outputs = Dense(18, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])


model.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("gru.h5", monitor="val_loss", keep_best_only=True)])
