import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM, Input, RepeatVector, TimeDistributed
from keras.layers import Embedding, Activation, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, EarlyStopping, ModelCheckpoint
import math
import matplotlib.pyplot as plt
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from dataloaders import test_loader
from load_pretrain import  read_data_filtered_augmented
import utils

X, y = read_data_filtered_augmented("../PreTrainingDataset", noise = False)

scaler = StandardScaler().fit(utils.flatten(X))
X_scaled = utils.scale(X, scaler)


# this has two goals, one is to test representation learning, and the other is
# to use the weights from the encoder for transfer learning, as done in
# https://arxiv.org/abs/1502.04681
# This model currently does not work
class LSTM_ae:
    def __init__(self,X):
        inputs = Input(shape=(None, X.shape[-1]))
        encoded = LSTM(40, activation = 'relu', return_sequences=True)(inputs)
        encoded = LSTM(20, activation = 'relu')(inputs)
        decoded = RepeatVector(X.shape[1])(encoded)
        encoded = LSTM(20, activation = 'relu', return_sequences=True)(inputs)
        decoded = LSTM(40, return_sequences=True, activation='relu')(decoded)
        decoded = TimeDistributed(Dense(X.shape[-1]))(decoded)
        self.sequence_autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)

    def fit(self,X, epochs = 1000, batch_size = 400, shuffle = True):
        cp = ModelCheckpoint(filepath="models/lstm_autoencoder.h5",
                               save_best_only=True,
                               verbose=0)
        tb = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)
        self.sequence_autoencoder.compile(optimizer = 'adam', loss = 'mse')
        self.sequence_autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks = [cp, tb])


ae = LSTM_ae(X_scaled)
ae.fit(X_scaled, epochs=1000)

embedding = ae.encoder.predict(X_scaled)

class manifold_in_embedding:
    def __init__(self, encoder,
                 umap_dim=2,
                 umap_neighbors=10,
                 umap_min_distance=float(0),
                 umap_metric='euclidean',
                 random_state=0):
        self.manifold = umap.UMAP(
            random_state=random_state,
            metric=umap_metric,
            n_components = umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_distance

        )
