import tensorflow.keras as keras
import numpy as np
import scipy.io
import abc
from .helpers import read_file_validation, pad_along_axis
from .augmentors import window_roll, roll_labels
from .loaders import Loader

# This function returns two lists. basically groups on subject, then exercise
# data: list of lists of 2D matrix (default emg data only)
# labs: list of lists of arrays of which exercise trial 
def _load_by_subjects_raw(nina_path=".", subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], options=None):
    data = []
    labs = []
    if type(subjects) is int:
        subs = [subjects]
    else:
        subs = subjects
    for sub in subs:
        subData = []
        subLabs = []
        for i in range(1,4):
            path = nina_path + "/ninaPro/" + "s" + str(sub) + "/S" + str(sub) + "_E" + str(i) + "_A1.mat"
            fileData, l = _load_file(path, options)
            subData.append(fileData)
            subLabs+=l
        data.append(subData)
        labs+= subLabs
    return data, labs


class NinaLoader(Loader):
    def __init__(self, path: str, excercises: list, process_fns: list, augment_fns: list, scale=False, step =5, window_size=52):
        self.path = path
        self.excercises = excercises
        if type(excercises) is not list:
            self.excercises = [excercises]
        self.processors = process_fns
        self.augmentors = augment_fns
        self.read_data()
        self.process_data()
        print(f"Shape of emg: {np.shape(self.emg)}")
        print(f"Shape of labels: {np.shape(self.labels)}")
        if augment_fns is not None:
            self.augment_data(step, window_size)
        self.emg = np.moveaxis(np.concatenate(self.emg,axis=0),2,1)
        if scale:
            self.emg = pp.scale(self.emg)
    
    # features can be an array if we need to pass back additional 
    # features with the emg data. could help recycle this
    # loader if we want to group by rerepetition later on.
    def _load_file(self, path, features=None):
        res = scipy.io.loadmat(path)
        data = []
        # Might need to start clipping emg segments here... RAM is 
        # struggling to keep up with massive sizes
        emg = res['emg'][:,:8]
        lab = res['restimulus'][:]
        data.append(emg)
        if features:
            for ft in features:
                sameDim = data[0].shape[0]==np.shape(res[ft])[0]
                newData = []
                if not sameDim and np.shape(res[ft])[1]==1:
                    newData = np.full((np.shape(data[0])[0],1), res[ft][0,0])
                else:
                    newData = res[ft]
                data.append(newData)
        return np.concatenate(data,axis=1), lab

    def _load_by_trial_raw(self, trial=1, options=None):
        data = []
        labs = []
        reps = []
        for i in range(1,11):
            print(f"Starting load of {i}/10 .mat files")
            path = self.path + "/ninaPro/" + "s" + str(i) + "/S" + str(i) + "_E" + str(trial) + "_A1.mat"
            fileData, l = self._load_file(path, options)
            data.append(fileData)
            labs.append(l)
            
        return data, labs

    def _read_group_to_lists(self):
        res = []
        labels = []
        for e in self.excercises:
            # In the papers the exercises are lettered not numbered
            # Also watchout, the 'exercise' col in each .mat are 
            # numbered weird. 
            # ex: /s1/S1_E1_A1.mat has says ['exercise'] is 3.
            # 1 ==> 3 | 2 ==> 1 | 3 ==> 2 (I think)[again only if reading 
            # column in raw .mat]
            if e == 'a':
                e = 1
            elif e == 'b':
                e = 2
            elif e == 'c':
                e = 3
            exData, l = self._load_by_trial_raw(trial=e)
            res+=exData
            labels+=l
        return np.concatenate(res), np.concatenate(labels)


    

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


# x = NinaLoader("../../PreTrainingDataset", [pp.butter_highpass_filter], [aa.add_noise])