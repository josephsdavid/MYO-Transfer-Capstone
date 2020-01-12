from keras.models import load_model
from dataloaders import test_loader
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
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import itertools as it


# thankful for grid search homework
pars = ["scale","noise","filter"]
tunables = {key: [True, False] for key in pars}
grid = list(it.product(*((tunables[i]) for i in tunables)))





import itertools
