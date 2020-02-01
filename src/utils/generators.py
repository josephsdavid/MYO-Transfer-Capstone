import tensorflow as tf
import numpy as np
from .loaders import *

class PreValGenerator(PreValidationLoader, tf.keras.utils.Sequence):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False,
                 batch_size=400, shuffle=True, step=5, window_size=52):
        # python is so fucking cool
        super(PreValGenerator, self).__init__(path, process_fns, augment_fns, scale, step, window_size)
        self.batch_size = batch_size
        self.shuffle =shuffle
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(self.emg.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.indexes=np.arange(self.emg.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.emg[indexes,:,:],  self.labels[indexes]



class PreTrainGenerator(PreTrainLoader, tf.keras.utils.Sequence):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False,
                 batch_size=400, shuffle=True, step=5, window_size=52):
        # python is so fucking cool
        super(PreTrainGenerator, self).__init__(path, process_fns, augment_fns, scale, step, window_size)
        self.batch_size = batch_size
        self.shuffle =shuffle
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(self.emg.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.indexes=np.arange(self.emg.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.emg[indexes,:,:],  self.labels[indexes]


