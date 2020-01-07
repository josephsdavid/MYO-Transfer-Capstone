import numpy as np

# load up the first example first!
#
trial = np.fromfile("../PreTrainingDataset/Male0/training0/classe_0.dat", dtype = np.int16)
try2 = trial.reshape(int(trial.shape[0]/8),8)
abs((np.vstack(np.split(trial, int(trial.shape[0]/8))) - try2)).sum()

try3 = read_file("../PreTrainingDataset/Male0/training0/classe_0.dat")

def read_file(path):
    f = np.array(np.fromfile(path, dtype = np.int16))
    example = []
    emg_vector = []
    for value in f:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (example == []):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
    return example
    #return f.reshape(int(f.shape[0]/8),8)


def read_group_to_lists(path, n_classes = 7):
    for candidate in range(12):
        mens = [read_file(path + '/Male' + str(candidate) + '/training0/classe_%d.dat' %i) for i in range(n_classes*4)]

    for candidate in range(7):
        womens = [read_file(path + '/Female' + str(candidate) + '/training0/classe_%d.dat' %i) for i in range(n_classes*4)]
    res = [mens] + [womens]
    return(res)

out = read_group_to_lists("../PreTrainingDataset")

import matplotlib.pyplot as plt

plt.plot(trial)

plt.show()
