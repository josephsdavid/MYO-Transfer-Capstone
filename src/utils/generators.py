import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from .loaders import *
from .ninaLoader import NinaLoader
#from .preprocessors import scale
def scale(arr3d):
    for i in range(arr3d.shape[0]):
        arr3d[i,:,:] /= np.abs(arr3d[i,:,:]).max(axis=0)
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
            rectify=False,
            step =5, window_size=52,
            batch_size=400, shuffle=True,
            validation=False,
            by_subject=False,
            sample_0=True):

        super(NinaGenerator, self).__init__(path, excercises,process_fns, augment_fns, scale, step, window_size)
        self.rectify=rectify
        self.scale=scale
        self.batch_size = batch_size
        self.shuffle =shuffle
        self._indexer(np.where(self.rep!=0))
        if sample_0:
            ids = np.where(self.labels==0)[0]
            ids2 = np.random.permutation(ids)
            #print(ids - ids2)
            #print(ids[0].shape[0] - ids[0][::18].shape[0])
            #ids2=tuple(ids[0][::18], )
            self._deleter(ids2[:-13000])
        v_subjects = np.array(np.unique(self.subject)[3:5])
        v_reps = np.array(np.unique(self.rep)[-2:])
        print("v_subjects: {}".format(v_subjects))
        print("v_reps: {}".format(v_reps))
        case_dict = {
                (False, False):np.where(np.isin(self.rep, v_reps, invert=True)),
                (True, False):np.where(np.isin(self.rep, v_reps)),
                (False, True):np.where(np.isin(self.subject, v_subjects, invert=True)),
                (True, True):np.where(np.isin(self.subject, v_subjects))
                }
        #print("number in rep 0: {}".format(self.emg[np.where(self.rep==0)].shape[0]))
        #print("number in label 0 : {}".format(self.emg[np.where(self.labels==0)].shape[0]))
        #print("number in label 1 : {}".format(self.emg[np.where(self.labels==1)].shape[0]))
        #print("number in label 2 : {}".format(self.emg[np.where(self.labels==2)].shape[0]))
        #print("number in label 3 : {}".format(self.emg[np.where(self.labels==3)].shape[0]))
        #print("number in label 4 : {}".format(self.emg[np.where(self.labels==4)].shape[0]))
        #print("number in label 5 : {}".format(self.emg[np.where(self.labels==5)].shape[0]))
        #print("number minlabel labe1 :: {}".format(self.emg[np.where(self.labels==lab10)].shape[0] / self.emg[np.where(self.rep!=1)].shape[0]))
        #print(1/18label1 :[(validation, by_subject)]
        # fix!!
        case = case_dict[(validation, by_subject)]
        self._indexer(case)
        print(self.emg.shape)
        self.on_epoch_end()



    def _indexer(self, id):
        self.emg = self.emg[id]
        self.rep = self.rep[id]
        self.labels=self.labels[id]
        self.subject=self.subject[id]

    def _deleter(self, id):
        self.emg = self.emg[~id]
        self.rep = self.rep[~id]
        self.labels=self.labels[~id]
        self.subject=self.subject[~id]



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
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)
        return out,  self.labels[indexes]

