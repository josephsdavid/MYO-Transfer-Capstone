import numpy as np
import scipy.io

guess = scipy.io.loadmat("../ninaPro/s1/S1_E1_A1.mat")

guess.keys()

guess['emg'].shape

guess['stimulus']

guess['emg'][0]

guess.keys()



def load_file(path):
    res = scipy.io.loadmat(path)
    emg = res['emg'][:,:8]
    print((res['stimulus'] - res['stimulus']).sum())
    return emg, res['restimulus']

emg, x = load_file("../ninaPro/s1/S1_E1_A1.mat")

def load_train_raw(path_to_nina = ".."):
    data = []
    labs = []
    for i in range(1,10):
        for j in range(1,4):
            path = path_to_nina + "/ninaPro/" + "s" + str(i) + "/S" + str(i) + "_E" + str(j) + "_A1.mat"
            emg, l = load_file(path)
            data.append(emg)
            labs.append(l*j)
    return(data, labs)



x, y = load_train_raw()

[o.shape for o in x]
# something weird
[o.max()-o.min() for o in y]

