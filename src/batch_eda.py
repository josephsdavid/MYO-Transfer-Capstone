import utils as u
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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



'''
make it into a transition plot!
samples[]:
    [0][3]
    [1][2]
'''
def plot_sample(sample):
    matplotlib.rcParams.update({'font.size': 22, 'font.weight':'bold'})
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=[
        'tomato',
        'coral',
        'indianred',
        'firebrick',
        'maroon',
        'darkred',
        'palevioletred',
        'darkmagenta',
        'blueviolet',
        'darkslateblue',
        'mediumpurple',
        'royalblue',
        'navy',
        'mediumblue',
        'dodgerblue',
        'deepskyblue'])
    fig, ax = plt.subplots(figsize=(12,12))
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for i in range(sample.shape[-1]):
        ax.plot(sample[:,i], label = str(i+1))
    #ax.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    #matplotlib.use('agg')
    #fig.savefig(path)
    return fig, ax
plt.style.use(['fivethirtyeight'])
matplotlib.rcParams.update({'font.size': 22, 'font.weight':'bold'})
matplotlib.use('agg')
for p, s in zip(paths, sample):
    f, ax = plot_sample(s)
    ax.set_facecolor('white')
    ax.set_title(p)
    f.savefig(f"fig/{p}.svg",format="svg", transparent=True)
#    plt.show()
matplotlib.use(b)

#rect = sample[2]
#periods = range(3,16)
#for p in range(len(periods)):
#    fig = plt.figure()
#    period = periods[p]
#    s = u.ma(rect, period)
#    for i in range(s.shape[-1]):
#        plt.plot(s[:,i])
#    plt.title(p+3)
#import pdb; pdb.set_trace()  # XXX BREAKPOINT
#plt.show()

# show augmentation
from matplotlib import cm
pre = sample[-1]
rdict = {x/2:[(x/2)%30]*((x//2)%30) for x in range(100)}
rlen = len(sum([[(x/2)%30]*((x//2)%30) for x in range(120)], []))
rfix = list(rdict.values())
rfix = list(set(sum(rfix, [])))
cm_nums = [2*len(rdict[i])/rlen for i in list(rdict.keys())[::2]]

cm_nums = np.array(cm_nums)
cm_nums /= cm_nums.max()
c = cm.inferno(cm_nums)
fig, ax = plt.subplots(figsize=(12,10))
for i in range(50):
    aug = u.add_noise_snr(pre, rfix[i])
    ax.plot(aug[:,1], label = str(i+1), color=c[i])
sm = plt.cm.ScalarMappable(cmap=cm.inferno, norm=plt.Normalize(vmin=0, vmax=0.033))
plt.colorbar(sm)
plt.savefig("fig/augmentation.png")
matplotlib.use(b)
