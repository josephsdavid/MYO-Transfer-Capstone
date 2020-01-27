from tensorflow.keras.models import load_model
from dataloaders import test_loader
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import read_data, read_data_augmented, read_data_filtered
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.layers import Embedding, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, targets, batch_size = 400, dim = (53), n_channels = 8, shuffle=True, n_classes=7):
        self.dim = dim
        self.batch_size = batch_size
        self.targets = targets
        self.features=features
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle =shuffle
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(self.features.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.indexes=np.arange(self.features.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.features[indexes,:,:],  self.targets[indexes]



x_tr = np.load("../data/x_train.npy")
y_tr = np.load("../data/y_train.npy")
x_val = np.load("../data/x_val.npy")
y_val = np.load("../data/y_val.npy")

# fiddle with batch_size
training_set = DataGenerator(x_tr, y_tr, batch_size = 400)
val_set=DataGenerator(x_val, y_val, batch_size = 400)


start = Input((None, 8), name = 'Input')
# or 170 or 298
x = LSTM(170, activation = 'tanh', dropout=0.2, recurrent_dropout=0.25)(start)
out = Dense(7, activation='softmax' )(x)

lstm = Model(start, out)

fit_options = {
    'epochs': 100,
    'verbose':1,
}
compilation_options = {
    'optimizer':optimizers.Adam(lr=0.0005),
    'loss' : 'sparse_categorical_crossentropy',
    'metrics' : ['accuracy']}

cb = EarlyStopping(monitor="val_loss", patience = 15)
cb2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
check = tf.keras.callbacks.ModelCheckpoint("simple_lstm.h5", monior='val_loss', save_best_only=True)

lstm.compile(**compilation_options)
history = lstm.fit(training_set,
                         validation_data=val_set,
                         **fit_options, callbacks = [cb, cb2, check])


plt.subplot(212)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2)
plt.savefig("simple_lstm_training.png")
plt.show()

#
#
######################### temporary evaluation
#X_test, y_test = test_loader("../EvaluationDataset")
#y_test = to_categorical(y_test)
#X_test.shape
#
#lstm.model.evaluate(X_test, y_test)
## 68% accuracy
#
#
#preds = lstm.model.predict(X_test)
#
#
#fig = plt.figure()
#for k in range(preds.shape[-1]):
#    ax = fig.add_subplot(3, 3, k+1)
#    ax.plot(np.linspace(0,1, 200),gaussian_kde(preds[:,k])(np.linspace(0,1,200)), label = k)
#    ax.set_title(str(k))
#plt.savefig('simple_lstm_class_probs.png')
#plt.show()
#
#
#fig = plt.figure()
#for k in range(preds.shape[-1]):
#    ax = fig.add_subplot(3, 3, k+1)
#    ax.plot(np.linspace(0,1, 200),gaussian_kde(y_test[:,k])(np.linspace(0,1,200)), label = k)
#    ax.set_title(str(k))
#plt.savefig('actual_class_probs.png')
#plt.show()
#
#
