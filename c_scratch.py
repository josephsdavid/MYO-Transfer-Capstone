#%%
from EDA.visualization import plot_trial, calculate_spectrogram_dataset, show_spectrogram
from Preprocessing import data_utils, filters, data2
import numpy as np

# %%
examples, labels = data2.read_data('EvaluationDataset',type='training0')
# plot_trial(examples, labels,0,1,(10,15))


# %%
# plot_trial(examples, labels, 7,1, (10,20), True, ptype='freq')
# plot_trial(examples, labels, 7,1, (10,20), True, ptype='ts')
# plot_trial(examples, labels, 7,1, (10,20), True, ptype='spec')