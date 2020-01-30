import tensorflow as tf
import math
from tensorflow.keras.models import load_model
from dataloaders import test_loader
from scipy.stats import gaussian_kde
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cycle import CyclicLR
import numpy as np
from load_pretrain import read_data, read_data_augmented, read_data_filtered
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, BatchNormalization, GaussianNoise, LSTM
from tensorflow.keras.layers import Embedding, Activation, PReLU, TimeDistributed, RepeatVector, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.regularizers as rr
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


print(x_val.shape, x_tr.shape)


# fiddle with batch_size
training_set = DataGenerator(x_tr, y_tr, batch_size = 3000, shuffle=True)
val_set=DataGenerator(x_val, y_val, batch_size = 3000, shuffle=False)


#inn = Input((52, 8))
#f = Flatten()(inn)
#encoded = Dense((300), activation='relu',
#               activity_regularizer=rr.l1(10e-5))(f)
#encoded = Dense((100), activation='relu',
#               activity_regularizer=rr.l1(10e-5))(encoded)
#encoded = Dense((40))(encoded)
#decoded = Dense(100)(encoded)
#decoded = Dense(300)(decoded)
#decoded = Dense((416), activation='sigmoid')(decoded)
#
#ae = Model(inn, decoded)
#ae.compile('adam','mse')
#ae.fit(x_tr, x_flat, batch_size = 3000, epochs = 100)
#
#
#x = RepeatVector(52)(encoded)
#x = LSTM(40
#         , recurrent_dropout=0.5,dropout=0.5, activation = 'tanh'
#         )(encoded)
#out = Dense(7, activation='softmax' )(x)
#
#sparseguy = Model(inn, out)
#sparseguy.compile('adam','sparse_categorical_crossentropy')
#sparseguy.fit(x_tr, y_tr, validation_data = (x_val, y_val), batch_size = 3000, epochs = 10)
#
#inn = Input((52, 8))



start = Input((60, 8), name = 'Input')
x = LSTM(16
         , recurrent_dropout=0.1,dropout=0.1
         , activation = 'tanh', return_sequences= True,
         )(start)
x = LSTM(32
         , recurrent_dropout=0.1,dropout=0.1
         , activation = 'tanh', return_sequences=True
         )(x)
x = LSTM(64
         , recurrent_dropout=0.1,dropout=0.1
         , activation = 'tanh',
         )(x)
out = Dense(7, activation='softmax' )(x)
lstm = Model(start, out)
fit_options = {
    'epochs': 1000,
    'verbose':1,
}
compilation_options = {
    'optimizer':optimizers.SGD(1e-7, momentum=0.9),
    'loss' : 'sparse_categorical_crossentropy',
    'metrics' : ['accuracy']}
cb = EarlyStopping(monitor="val_loss", patience = 30)
#cb2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose = 1)
clr = CyclicLR(
	mode="triangular",
	base_lr=1e-7,
	max_lr=1e-2,
	step_size= 8 * (len(training_set)))
check = tf.keras.callbacks.ModelCheckpoint("simple_lstm.h5", monior='val_loss', save_best_only=True)
lstm.compile(**compilation_options)
print(lstm.summary())
# run through entire set every 100 epochs
# this lets us trick the reduce lr on plateau into updating dynamically
history = lstm.fit(training_set, steps_per_epoch = int(len(training_set)/8),#batch_size = 6000,
                   validation_data = val_set,**fit_options, callbacks = [cb, clr, check])



plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
