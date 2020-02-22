import utils as u
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
b = matplotlib.get_backend()

batch=512
train = u.NinaGenerator("../data/ninaPro", ['b'], None,
                        None, validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, window_size=52, shuffle=False)

x_batch = train[30][0]

sample = []
sample += [x_batch[150]]
sample += [u.butter_highpass_filter(sample[-1])]
sample += [np.abs(sample[-1])]
sample += [u.ma(sample[-1], 15)]
paths = ['Raw sEMG', 'Highpass Filtered sEMG','Rectified Filtered sEMG', 'Smoothed Rectified sEMG']

def plot_sample(sample):
    fig, ax = plt.subplots()
    for i in range(sample.shape[-1]):
        ax.plot(sample[:,i], label = str(i+1))
    ax.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    #matplotlib.use('pdf')
    #fig.savefig(path)
    return fig, ax

matplotlib.use('pdf')
for p, s in zip(paths, sample):
    f, ax = plot_sample(s)
    ax.set_title(p)
    f.savefig(f"fig/{p}.pdf")
matplotlib.use(b)
