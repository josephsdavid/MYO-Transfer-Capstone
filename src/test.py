import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

batch=400

min_lr = 1e-7
max_lr = 1e-2

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch)

print(len(train_set))
print(train_set.__getitem__(0)[0].shape)
print(len(val_set))
print(val_set.__getitem__(0)[0].shape)

lr_manager = cb.OneCycleLR(1e-3,
                           end_percentage = 0.1, scale_percentage = None,
                           maximum_momentum = 0.95, minimum_momentum=0.85
                           )

inputs = Input((52, 8))
x = LSTM(40, activation = 'tanh', dropout=0.1, recurrent_dropout=0.1)(inputs)
outputs = Dense(7, activation='softmax')(x)

lstm = Model(inputs, outputs)
optim = SGD(lr=0.0025, momentum = 0.95, nesterov = True)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics = ['accuracy'])

history = lstm.fit(train_set, epochs=100, validation_data=val_set, callbacks=[lr_manager])


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
plt.savefig("results/simple_lstm_training.png")
plt.show()


