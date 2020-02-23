import numpy as np
import torch
import torch.nn as nn
import utils as u
batch=512

train = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
test = u.NinaMA("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class simple_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
