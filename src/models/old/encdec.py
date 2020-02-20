import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import n2d

latent_dim=256

batch=1000
clr=cb.OneCycleLR(
                 max_lr=0.01,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


def holder(a,b):
	return a, b
train_set = u.PreTrainGenerator("../EvaluationDataset",
                                [u.butter_highpass_filter],
				[holder], batch_size = batch, scale=False
                                )
print(len(train_set))
val_set = u.PreValGenerator("../PreTrainingDataset",
                            [u.butter_highpass_filter],
			    [holder], batch_size = batch, scale=False
                            )


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
	inputs = Input(shape=(52, 8))
	o, h0, c0  = LSTM(400, activation = 'tanh', return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(inputs)
	o2, h1, c1 = LSTM(200, activation = 'tanh', return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(o)
	encoder = LSTM(latent_dim, return_state=True, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
	encoder_outputs, state_h, state_c = encoder(o2)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]
	decoded = RepeatVector(52)(encoder_outputs)

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh', dropout=0.5, recurrent_dropout=0.5)
	decoder_outputs, _, _ = decoder_lstm(decoded,
			                                     initial_state=encoder_states)
	decoder_outputs =LSTM(200, activation='tanh', return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(decoder_outputs, initial_state=[h1,c1])
	decoder_outputs =LSTM(400, activation='tanh', return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(decoder_outputs, initial_state=[h0,c0])
	decoder_dense = (Dense(7, activation='softmax'))
	decoder_outputs = decoder_dense(decoder_outputs)

	lstm = Model(inputs, decoder_outputs)
	optim = SGD(clipnorm=1., momentum=0.9, nesterov=True)
	lstm.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


stopper = EarlyStopping(monitor = "val_loss", patience=20)
history = lstm.fit(train_set, epochs=100, validation_data=val_set, callbacks=[stopper, clr], workers=16, use_multiprocessing=True)
lstm.save("result/fancy_training.h5")
import joblib
joblib.dump(history, "result/best_history.obj")

