import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from .loaders import *
from .ninaLoader import NinaLoader
#from .preprocessors import scale
def scale(arr3d):
    for i in range(arr3d.shape[0]):
        arr3d[i,:,:] /= arr3d[i,:,:].max(axis=0)
    return arr3d


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







class NinaGenerator(NinaLoader, tf.keras.utils.Sequence):
    def __init__(self, path: str, excercises: list,
            process_fns: list,
            augment_fns: list,
            scale=False,
            step =5, window_size=52,
            batch_size=400, shuffle=True,
            validation=False,
            by_subject=False):
        super(NinaGenerator, self).__init__(path, excercises,process_fns, augment_fns, scale, step, window_size)
        self.scale=scale
        self.batch_size = batch_size
        self.shuffle =shuffle
        print(min(self.labels))
        print(max(self.labels))
        self._indexer(np.where(self.rep!=0))
        v_subjects = np.array((9, 10, 11))
        v_reps = np.array((4,5,6))
        case_dict = {
                (False, False):np.where(np.isin(self.rep, v_reps, invert=True)),
                (True, False):np.where(np.isin(self.rep, v_reps)),
                (False, True):np.where(np.isin(self.subject, v_subjects, invert=True)),
                (True, True):np.where(np.isin(self.subject, v_subjects))
                }
        case=case_dict[(validation, by_subject)]
        # fix!!
        self._indexer(case)
        self.act = np.where(self.rep==0)
        self.on_epoch_end()



    def _indexer(self, id):
        self.emg = self.emg[id]
        self.rep = self.rep[id]
        self.labels=self.labels[id]
        self.subject=self.subject[id]

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
        out = self.emg[indexes,:,:]
        if self.augmentors is not None:
            for f in self.augmentors:
                for i in range(out.shape[0]):
                    out[i,:,:]=f(out[i,:,:])
        if self.scale:
            out = scale(out)
        return out,  self.labels[indexes]

