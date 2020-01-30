import tensorflow as tf
import numpy as np
import loaders as l
import preprocessors as pp
import augmentors as aa

class val_generator(l.ValidationLoader, tf.keras.utils.Sequence):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False, batch_size=400, shuffle=True):
        # python is so fucking cool
        super(val_generator, self).__init__(path, process_fns, augment_fns, scale)
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


x = val_generator("../../PreTrainingDataset", [pp.butter_highpass_filter], [aa.add_noise])

print(x.__getitem__(0)[0].shape)
