import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import callbacks as cb
import utils as u
from builders import build_att_gru
batch=512

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def ma(window, n):
    return np.vstack([moving_average(window[:,i], n) for i in range(window.shape[-1])]).T

def ma_batch(batch, n):
        return np.dstack([ma(batch[i,:,:], n) for i in range(batch.shape[0])])


def scale(arr3d):
    for i in range(arr3d.shape[0]):
        arr3d[i,:,:] /= np.abs(arr3d[i,:,:]).max(axis=0)
    return arr3d


class NinaMA(u.NinaGenerator):
    def __init__(self,
        path: str,
        excercises: list,
        process_fns: list,
        augment_fns: list,
        scale=False,
        rectify=False,
        step=5,
        window_size=52,
        batch_size=400, shuffle=True,
        validation=False,
        by_subject=False,
        sample_0=True,
        n=5, super_augment = False):
        super().__init__(
        path,
        excercises,
        process_fns,
        augment_fns,
        scale,
        rectify,
        step,
        window_size,
        batch_size, shuffle,
        validation,
        by_subject,
        sample_0
        )
        self.n = n
        self.super_augment = super_augment
        if super_augment:
            if self.augmentors is not None:
                o = [self.emg]
                ls = [self.labels]
                rs = [self.rep]
                ss = [self.subject]
                for f in self.augmentors:
                    inn = [self.emg[i,:,:] for i in range(self.emg.shape[0]) if self.labels[i] != 0]
                    ls.append(self.labels[np.where(self.labels != 0)])
                    rs.append(self.rep[np.where(self.labels != 0)])
                    ss.append(self.subject[np.where(self.labels != 0)])
                    with multiprocessing.Pool(None) as p:
                        res = p.map(f, inn)
                    o.append(np.moveaxis(np.dstack(res), -1, 0))
                self.emg = np.concatenate(o, axis=0)
                self.labels = np.concatenate( ls, axis=0)
                self.rep = np.concatenate( rs, axis=0)
                self.subject = np.concatenate( ss, axis=0)
                print(self.labels.shape)
                print(self.rep.shape)
                print(self.subject.shape)
                print(self.emg.shape)
                print("okokok")
                self.on_epoch_end()
        self.labels = to_categorical(self.labels)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:]
        if not self.super_augment:
            if self.augmentors is not None:
                for f in self.augmentors:
                    for i in range(out.shape[0]):
                        if self.labels[i,0] == 1:
                            out[i,:,:] = out[i,:,:]
                        else:
                            out[i,:,:]=f(out[i,:,:])
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)
        return np.moveaxis(ma_batch(out, self.n), -1, 0),  self.labels[indexes,:]


''' begin of analysis in earnest '''



