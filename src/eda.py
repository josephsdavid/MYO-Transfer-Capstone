from importlib import reload
import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

batch = 1280



train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
       None, validation=False, by_subject = False, batch_size=batch, scale = True)


#print(train.act)
#
#print(train.emg.shape)
train.act


nplots = np.unique(train.subject).shape[0]
#        self.flat = [self.emg, self.labels, self.rep, self.subject]
sub = np.vstack(train.flat[-1]).flatten()
rep = np.vstack(train.flat[-2]).flatten()
lab = np.vstack(train.flat[1]).flatten()
emg = np.vstack(train.flat[0])

# plot all labels
for l in np.unique(lab):
    print(str(l))
    other=np.where(lab==0)
    rest = np.where((lab==l))
    #rest = np.where(lab==l)
    print(rest)
    #emg[other]=0.
    sub1, emg1, lab1 = (x[rest] for x in [sub, emg, lab])
    fig = plt.figure(figsize=(20,30))
    for i in range(np.unique(sub1).shape[0]):
        ax = fig.add_subplot(np.unique(sub1).shape[0],2, i+1)
        #c = cm.Paired(i/len(imps.keys()), 1)
        ax.plot(emg1[np.where(sub1==np.unique(sub1)[i])])
        ax.set_title(i)
    #fig.tight_layout()
    fig.suptitle("label: {}".format(l), fontsize=48)
    #plt.show()
    plt.savefig("{}.png".format(l) ,bbox_inches="tight")
    plt.clf()


