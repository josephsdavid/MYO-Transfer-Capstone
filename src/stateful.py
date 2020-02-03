import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


batch=35


train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
                                [u.add_noise], batch_size = batch,
                                step=1, window_size=15, shuffle=False)
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
                            [u.add_noise], batch_size = batch,
                            step=1, window_size=15, shuffle=False)

dropout = 0.028638817753399493
rec_drop=0.14330185842693177
inputs = Input(batch_shape=(batch, 15, 8))
#x = LSTM(450, activation = 'tanh',
#		dropout=dropout, recurrent_dropout=rec_drop, stateful=True, return_sequences=True)(inputs)
#x = LSTM(150, activation = 'tanh',
#		dropout=dropout, recurrent_dropout=rec_drop, stateful=True, return_sequences=True)(x)
x = LSTM(20, activation = 'tanh',
		dropout=0.5, recurrent_dropout=0.5, stateful=True)(inputs)
outputs = Dense(7, activation='softmax')(x)
lstm = Model(inputs, outputs)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics= ['accuracy'])

check = ModelCheckpoint("result/stateful_lstm.h5", monitor="val_loss", save_best_only=True)


stopper = EarlyStopping(monitor="val_loss", patience=20)

class ResetStatesCallback(Callback):
	def on_batch_end(self, batch, logs={}):
		self.model.reset_states()
loss = 0
accuracy=[]
for e in range(len(train_set)*4):
	print("epoch: {}/{}".format(e, len(train_set)*4))
	x, y = train_set.__getitem__(e%len(train_set))
	res = lstm.train_on_batch(x, y)
	loss += res[0]
	print(loss/((e+1)%1000))
	accuracy.append(res[1])
	print("accuracy: {}".format(sum(accuracy) / len(accuracy)))
	if e%100==True: lstm.reset_states()
	if (e%1000==0 and e != 0):
		print(loss/10000)
		print("accuracy: {}".format(sum(accuracy) / len(accuracy)))
		loss=0
		accuracy=[]
		lstm.evaluate(val_set, steps=10000)
		lstm.save("result/stateful_lstm.h5")

lstm.save("result/stateful_lstm.h5")
