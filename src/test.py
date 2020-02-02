import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

batch=50

min_lr = 1e-7
max_lr = 1e-2



strategy = tf.distribute.MirroredStrategy()

def holder(a,b):
	return a, b

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = batch
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = batch)

print(len(train_set))
print(train_set.__getitem__(0)[0].shape)
print(len(val_set))
print(val_set.__getitem__(0)[0].shape)

with strategy.scope():
	inputs = Input((52,8))
	ls = Bidirectional(LSTM(20, activation="tanh", dropout=0.5,
		recurrent_dropout=0.5, return_sequences=True, return_state=True))
	x, h1, c1 = ls(inputs)
	ls2 = Bidirectional(LSTM(20, activation="tanh", dropout=0.5, recurrent_dropout=0.5,
			return_sequences=True, return_state=True
			))
	x, h, c = ls2(x,initial_state = [h1,c1])
	x = Bidirectional(LSTM(20, activation="tanh", dropout=0.5, recurrent_dropout=0.5
			)(x,initial_state = [h,c]))
	outputs = Dense(7, activation='softmax')(x)
	lstm=Model(inputs, outputs)
	# clip your gradients!!
	optim = Adam(clipnorm=1.)
	lstm.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics = ['accuracy'])
	# https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html


stopper = EarlyStopping(monitor = "val_loss", patience=20)
history = lstm.fit(train_set, epochs=100, validation_data=val_set, callbacks=[stopper], workers=16, use_multiprocessing=True, steps_per_epoch = len(train_set)//10, validation_steps = len(val_set)//2)
import joblib
joblib.dump(history, "result/best_history.obj")
lstm.save("result/fancy_training.h5")

