#%%
from keras.models import load_model
from dataloaders import test_loader, val_loader
from scipy.stats import gaussian_kde
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from load_pretrain import  read_data_augmented
from keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from keras.layers import Embedding, Activation
from keras.models import Model
from keras.applications import Xception
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import itertools as it

import json
import os
import argparse

#%%
#########################################
############ CONFIG GPUS ################
#########################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())
 
# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

#%%
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#%%
class simple_lstm_classifier:
    def __init__(self, X, y, act = 'tanh', dropout = 0, stateful = False, 
            compilation_options = {
                'loss' : 'categorical_crossentropy',
                'metrics' : ['accuracy']}):
    
        n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

        adam = optimizers.adam(lr = 0.0005)
        
        # check to see if we are compiling using just a single GPU
        
        
        start = Input((None, n_features), name = 'Input')
        x = LSTM(20, activation = act, name = 'LSTM_1', stateful = stateful, dropout = dropout, recurrent_dropout = dropout)(start)
        out = Dense(n_outputs, activation = 'softmax')(x)
        
        self.model = Model(start, out)
        
        # initialize the optimizer and model
        print("[INFO] compiling model...")
        self.model.compile(optimizer = adam, **compilation_options)
        

    def fit(self, X, y, lr = .005,
            fit_options = {
                'epochs': 100,
                'batch_size':400,
                'validation_split':0.33,
                'shuffle':False,
                'verbose':1
            }):
        # train the network
        print("[INFO] training network...")

        self.history = self.model.fit(X, y, **fit_options)

NUM_WORKERS = 2

fit_options = {
    'epochs': 50,
    'batch_size':500,
    'shuffle':False,
    'verbose':1
}
#%%
# thankful for grid search homework
pars = ["scale"]
tunables = {key: [True, False] for key in pars}
grid = list(it.product(*(tunables[i] for i in tunables)))

results = {}
for i in range(len(grid)):
    hp = dict(zip(pars, grid[i]))
    X_test, y_test = test_loader("../EvaluationDataset", scale = grid[i][0])
    y_test = to_categorical(y_test)
    X_val, y_val = val_loader("../EvaluationDataset", scale = grid[i][0])
    y_val = to_categorical(y_val)
    X, y = read_data_augmented("../PreTrainingDataset", **hp)
    y = to_categorical(y)
    
    lstm = simple_lstm_classifier(X,y, dropout = 0.1)
    lstm.fit(X, y, lr = 0.0005, fit_options=fit_options)
    score = lstm.model.evaluate(X_val, y_val)[1]
    score0=lstm.model.evaluate(X_test, y_test)[1]
    results[grid[i]] = {"val":score,"test":score0}
    with open(os.path.join(__location__,'results_multi.txt'), 'w') as f:
        print(results , file=f,)
results

# with open(os.path.join(__location__,'results_multi.txt'), 'w') as f:
#     print(results, file=f)
    #  file.write(json.dumps(_res)) # use `json.loads` to do the reverse


# %%
