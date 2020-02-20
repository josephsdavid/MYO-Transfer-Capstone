import utils as u
import multiprocessing
import numpy as np
import callbacks as cb
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from builders import build_att_gru
batch=512


import pdb; pdb.set_trace()  # XXX BREAKPOINT


train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)


n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

print("n_timesteps{}".format(n_time))

neg = Constant(value=-1)

model=build_att_gru(n_time, n_class, learning_rate=0.01)
tf.keras.utils.plot_model(model, to_file="attn.png", show_shapes=True, expand_nested=True)
model.compile(Ranger(), loss='categorical_crossentropy', metrics=['accuracy'])
class_weights = {i:1/(n_class) if i==0 else 1 for i in range(1, n_class+1)}
h2 = model.fit(train, epochs=100, validation_data=test, shuffle=False, callbacks=[ModelCheckpoint("gru2.h5", monitor="val_loss", keep_best_only=True), ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)], use_multiprocessing=True, workers=12, class_weight=class_weights)


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
plt.savefig("simple_lstm_training.png")
