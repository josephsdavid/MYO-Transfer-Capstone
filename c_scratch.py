#%%
from EDA.visualization import plot_trial, calculate_spectrogram_dataset, show_spectrogram
from Preprocessing import data_utils, filters
import numpy as np

# %%
examples, labels = data_utils.read_data('PreTrainingDataset',type='training0')
# plot_trial(examples, labels,0,1,(10,15))


# %%
plot_trial(examples, labels, 2,1, (10,20), True, ptype='freq')
plot_trial(examples, labels, 2,1, (10,20), True, ptype='ts')
plot_trial(examples, labels, 2,1, (10,20), True, ptype='spec')
# %%



# %%
