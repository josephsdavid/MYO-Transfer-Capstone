import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

batch=3000
clr=cb.OneCycleLR(
                 max_lr=.1,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)
min_lr = 1e-7
max_lr = 1e-2



strategy = tf.distribute.MirroredStrategy()

def holder(a,b):
	return a, b

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch, scale=False
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = batch, scale=False)

print(len(train_set))
print(train_set.__getitem__(0)[0].shape)
print(len(val_set))
print(val_set.__getitem__(0)[0].shape)

with strategy.scope():
	inputs = Input((52,8))
	x = inputs
	drop=0.028638817753399493
	rec_drop = 0.14330185842693177
	units=[450, 150, 400]
	seq=[True, True, False]
	for i in range(3):
		x = LSTM(units[i], activation="tanh", dropout=drop, recurrent_dropout=rec_drop, return_sequences=seq[i], name = "lstm_{}".format(i))(x)
	outputs = Dense(7, activation='softmax')(x)
	lstm = Model(inputs, outputs)

#	ls = (LSTM(450, activation="tanh", dropout=0.5,
#		recurrent_dropout=0.5, return_sequences=True, return_state=True))
#	x, h1, c1 = ls(inputs)
#	ls2 = (LSTM(450, activation="tanh", dropout=0.5, recurrent_dropout=0.5,
#			return_sequences=True, return_state=True
#			))
#	x, h, c = ls2(x,initial_state = [h1,c1])
#	x = (LSTM(450, activation="tanh", dropout=0.5, recurrent_dropout=0.5
#			)(x,initial_state = [h,c]))
#	outputs = Dense(7, activation='softmax')(x)
#	lstm=Model(inputs, outputs)
#	# clip your gradients!!
	optim = SGD(momentum=0.9, nesterov=True)
	lstm.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics = ['accuracy'])
#	# https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

print(lstm.summary())
stopper = EarlyStopping(monitor = "val_loss", patience=100)
history = lstm.fit(train_set, epochs=100, validation_data=val_set, callbacks=[stopper, clr], workers=16, use_multiprocessing=True, steps_per_epoch=len(train_set)//4)
lstm.save("result/source_net")

