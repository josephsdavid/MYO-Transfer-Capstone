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

batch=512

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


x_tr = []
y_tr = []
for t in train_set:
    x_tr.append(t[0])
    y_tr.append(t[1])
x_val = []
y_val = []
for t in val_set:
    x_val.append(t[0])
    y_val.append(t[1])

import random
temp = list(zip(x_tr, y_tr))
random.shuffle(temp)
x_tr, y_tr = zip(*temp)
temp = list(zip(x_val, y_val))
random.shuffle(temp)
x_val, y_val = zip(*temp)
x_tr = np.concatenate(x_tr,0).astype(np.float16)
y_tr = np.concatenate(y_tr,0).astype(np.float16)
x_val = np.concatenate(x_val,0).astype(np.float16)
y_val = np.concatenate(y_val,0).astype(np.float16)
print(x_tr.shape)

latent_dim=100

inputs = Input(shape=(52, 8))
o, h0, c0  = LSTM(400, activation = 'tanh', return_sequences=True, return_state=True)(inputs)
o2, h1, c1 = LSTM(200, activation = 'tanh', return_sequences=True, return_state=True)(o)
encoder = LSTM(latent_dim, return_state=True, activation='tanh')
encoder_outputs, state_h, state_c = encoder(o2)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
decoded = RepeatVector(52)(encoder_outputs)

# Set up the decoder, using `encoder_states` as initial state.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh')
decoder_outputs, _, _ = decoder_lstm(decoded,
		                                     initial_state=encoder_states)
decoder_outputs =LSTM(200, activation='tanh', return_sequences=True)(decoder_outputs, initial_state=[h1,c1])
decoder_outputs =LSTM(400, activation='tanh', return_sequences=True)(decoder_outputs, initial_state=[h0,c0])
decoder_dense = TimeDistributed(Dense(8))
decoder_outputs = decoder_dense(decoder_outputs)

ae = n2d.autoencoder_generator((inputs, encoder_outputs, decoder_outputs))

n_clusters=7
umapgmm=n2d.UmapGMM(n_clusters, umap_dim=100)
model=n2d.n2d(ae, umapgmm)
model.fit(x_tr, epochs=1000, batch_size=512, patience=100, weight_id='result/n2d.h5', optimizer=SGD(learning_rate=0.001, clipnorm=1., clipvalue=0.5, momentum = 0.9, nesterov=True))
preds=model.predict(x_val)


print(model.assess(y_val))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

def fit(obj, x,y, clf):
	embedding =  obj.encoder.predict(x)
	manifold= obj.manifolder.transform(embedding)
	clf.fit(manifold, y)

fit(model, x_tr, y_tr, rf)

def predict(obj, x, y, clf):
	embedding =  obj.encoder.predict(x)
	manifold= obj.manifolder.transform(embedding)
	return clf.predict(manifold)
rfp = predict(model, x_val, y_val, rf)

from sklearn.metrics import accuracy_score

print(accuracy_score(rfp, y_val))
