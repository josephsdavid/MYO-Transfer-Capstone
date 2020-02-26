import numpy as np
import utils as u
import builders.twostage as b
from optimizers import Ranger
import callbacks as cb
import losses as l
from tensorflow.keras.callbacks import ModelCheckpoint

batch = 512
# load data
train = u.NinaMA("../data/ninaPro", ['b','c'], [np.abs, u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = False, batch_size=batch,
                        scale = False, rectify=False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)
val = u.NinaMA("../data/ninaPro", ['b','c'], [np.abs, u.butter_highpass_filter],
                       None, validation=True, by_subject = False, batch_size=batch,
                       scale = False, rectify =False, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

# generate unseen test data from validation set hidden state:
test = u.TestGen(*val.test_data, shuffle=False, batch_size=batch)
zero_conditioner = u.TestGen(*val.test_data, shuffle=False, batch_size=batch, zeros=True)
zero_validator = u.TestGen(*val.test_data, shuffle=False, batch_size=batch, zeros=False)
import pdb; pdb.set_trace()  # XXX BREAKPOINT

# needed to build the model
n_time = train[0][0].shape[1]
n_class =train[0][1].shape[-1]

# needed to compile the model
loss = l.focal_loss( gamma=4.)
optim = Ranger()

# simple two stage feed_forward attention model
# has 128 cell linear domain layer and then the rest of the layers are defined
# with a list.
model = b.TwoStageAtt(n_time, n_class)
model.compile(optimizer = optim, loss = loss, metrics = ['accuracy'])

# named arguments to pretrain:
pretrain_kwargs = {
    'epochs' : 25, # toy example, but models converge quickly
    'validation_data' : val,
    'shuffle' : False, # the generator class shuffles the data much faster
    'callbacks': [
        ModelCheckpoint('two-stage.h5', monitor='val_loss', keep_best_only = True, save_weights_only = True),
        cb.CosineAnnealingScheduler(T_max = 100, eta_max = 1e-3, eta_min = 5e-5, epoch_start=5)
    ]
}

# pretrain with domain adaptation layer frozen
h = model.pretrain(train, **pretrain_kwargs)

# final fit for a few epochs then predict, is this not cheating????
adapt_kwargs = {
    'epochs' : 5,
    'shuffle' : False, # the generator class shuffles the data much faster
    'callbacks': [
        ModelCheckpoint('two-stage-part 2.h5', monitor='val_loss', keep_best_only = True, save_weights_only = True),
        cb.CosineAnnealingScheduler(T_max = 100, eta_max = 1e-3, eta_min = 5e-5, epoch_start=5)
    ],
    'validation_data':zero_validator
}

# fit with domain layer unfrozen everything else unfrozen
# is this cheating?
h2 = model.adapt(zero_conditioner, **adapt_kwargs)

model.model.evaluate(test)

import pdb; pdb.set_trace()  # XXX BREAKPOINT

