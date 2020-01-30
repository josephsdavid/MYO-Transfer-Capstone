import tensorflow.keras as keras
import numpy as np
import abc
from helpers import read_file_validation, pad_along_axis
import preprocessors as pp
import augmentors as aa

class Loader(abc.ABC):
    @abc.abstractmethod
    def read_data(self):
        pass

    @abc.abstractmethod
    def process_data(self):
        pass

    @abc.abstractmethod
    def augment_data(self):
        pass



class ValidationLoader(Loader):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False):
        print("ok")
        self.path = path
        self.processors = process_fns
        self.augmentors = augment_fns
        self.read_data()
        self.process_data()
        self.augment_data()
        self.emg = np.moveaxis(np.concatenate(self.emg,axis=0),2,1)
        if scale:
            self.emg = pp.scale(self.emg)

    def _read_group_to_lists(self):
        # grab in all the male candidates
        n_classes=7
        res = []
        labels = []
        trials = range(n_classes*4)
        for candidate in range(12):
            man = [read_file_validation(self.path + '/Male' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
            # list addition is my new favorite python thing
            labs = [t % n_classes for t in trials]
            res += man
            labels += labs

        # and all the female candidates
        for candidate in range(7):
            woman = [read_file_validation(self.path + '/Female' + str(candidate) + '/training0/classe_%d.dat' %i) for i in trials]
            labs = [t % n_classes for t in trials]
            res += woman
            labels += labs

        return res, labels

    def read_data(self):
        self.emg, self.labels = self._read_group_to_lists()
        self.emg = [pad_along_axis(x, 1000) for x in self.emg]

    def process_data(self):
        for f in self.processors:
            self.emg = [f(x) for x in self.emg]

    def augment_data(self):
        for f in self.augmentors:
            self.emg, self.labels = f(self.emg, self.labels)

        self.emg = [aa.window_roll(x, 5, 52) for x in self.emg]
        self.labels = aa.roll_labels(self.emg, self.labels)


def holder(x,y):
    return x, y


# x = ValidationLoader("../../PreTrainingDataset", [pp.butter_highpass_filter], [aa.add_noise])



