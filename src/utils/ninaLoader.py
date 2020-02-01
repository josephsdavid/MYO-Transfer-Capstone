import tensorflow.keras as keras
import numpy as np
import scipy.io
import abc
from .helpers import read_file_validation, pad_along_axis
from .augmentors import window_roll, roll_labels
from .loaders import Loader

# make file into 11-col (0-7: emg, 8: restimulus, 9: rerepetition, 10: subject)
def _load_file(path, features=None):
    res = scipy.io.loadmat(path)
    data = []
    emg = res['emg'][:,:8]
    data.append(emg)
    if features==None:
        features = ['restimulus', 'rerepetition', 'subject']

    for ft in features:
        sameDim = data[0].shape[0]==np.shape(res[ft])[0]
        newData = []
        if not sameDim and np.shape(res[ft])[1]==1:
            newData = np.full((np.shape(data[0])[0],1), res[ft][0,0])
        else:
            newData = res[ft]
        data.append(newData)
    return np.concatenate(data,axis=1)

def _load_by_trial_raw(nina_path = ".", trial=1, options=None):
    data = []
    labs = []
    reps = []
    for i in range(1,11):
        path = nina_path + "/ninaPro/" + "s" + str(i) + "/S" + str(i) + "_E" + str(trial) + "_A1.mat"
        fileData = _load_file(path, options)
        data.append(fileData)
    return data


def _load_by_subjects_raw(nina_path=".", subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], options=None):
    data = []
    if type(subjects) is int:
        subs = [subjects]
    else:
        subs = subjects
    for sub in subs:
        subData = []
        for i in range(1,4):
            path = nina_path + "/ninaPro/" + "s" + str(sub) + "/S" + str(sub) + "_E" + str(i) + "_A1.mat"
            fileData = _load_file(path, options)
            subData.append(fileData)
        data.append(subData)
    return data


class NinaLoader(Loader):
    def __init__(self, path: str, process_fns: list, augment_fns: list, scale=False, step =5, window_size=52):
        self.path = path
        self.processors = process_fns
        self.augmentors = augment_fns
        self.read_data()
        self.process_data()
        self.augment_data(step, window_size)
        self.emg = np.moveaxis(np.concatenate(self.emg,axis=0),2,1)
        if scale:
            self.emg = pp.scale(self.emg)
    
    def _read_group_to_lists(self):
        res = []
        labels = []
        trials = range(7*4)
        for instance in ['training0', 'Test0', 'Test1']:
            for candidate in range(15):
                man = [read_file_validation(self.path + '/Male' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
                # list addition is my new favorite python thing
                labs = [t % 7 for t in trials]
                res += man
                labels += labs
                # and all the female candidates
            for candidate in range(2):
                woman = [read_file_validation(self.path + '/Female' + str(candidate) + '/' + instance + '/classe_%d.dat' %i) for i in trials]
                labs = [t % 7 for t in trials]
                res += woman
                labels += labs
        return res, labels

    def read_data(self):
        self.emg, self.labels = self._read_group_to_lists()
        self.emg = [pad_along_axis(x, 1000) for x in self.emg]

    def process_data(self):
        for f in self.processors:
            self.emg = [f(x) for x in self.emg]

    def augment_data(self, step, window_size):
        for f in self.augmentors:
            self.emg, self.labels = f(self.emg, self.labels)

        self.emg = [window_roll(x, step, window_size) for x in self.emg]
        self.labels = roll_labels(self.emg, self.labels)
