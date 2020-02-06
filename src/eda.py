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


class NinaGenerator(u.NinaLoader, tf.keras.utils.Sequence):
    def __init__(self, path: str, excercises: list,
            process_fns: list,
            augment_fns: list,
            scale=False,
            step =5, window_size=52,
            batch_size=400, shuffle=True,
            validation=False,
            by_subject=False):
        super(NinaGenerator, self).__init__(path, excercises,process_fns, augment_fns, scale, step, window_size)
        self.scale=scale
        self.batch_size = batch_size
        self.shuffle =shuffle
        act = np.where(self.rep!=0)
        self._indexer(np.where(self.rep!=0))
        v_subjects = np.array((9,10))
        v_reps = np.array((4,5,6))
        case_dict = {
                (False, False):np.where(np.isin(self.rep, v_reps, invert=True)),
                (True, False):np.where(np.isin(self.rep, v_reps)),
                (False, True):np.where(np.isin(self.subject, v_subjects, invert=True)),
                (True, True):np.where(np.isin(self.subject, v_subjects))
                }
        case=case_dict[(validation, by_subject)]
        self._indexer(case)
        self.on_epoch_end()



    def _indexer(self, id):
        self.emg = self.emg[id]
        self.rep = self.rep[id]
        self.labels=self.labels[id]
        self.subject=self.subject[id]

    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(self.emg.shape[0]/self.batch_size))

    def on_epoch_end(self):
        self.indexes=np.arange(self.emg.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:]
        if self.augmentors is not None:
            for f in self.augmentors:
                for i in range(out.shape[0]):
                    out[i,:,:]=f(out[i,:,:])
        if self.scale:
            out = u.scale(out)
        return out.reshape((out.shape[0], -1),order='f'), out.reshape((out.shape[0], -1),order='f')


#strategy = tf.distribute.MirroredStrategy()
clr=cb.OneCycleLR(
                 max_lr=0.4,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85,
                 verbose=True)


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
rest = np.where(lab==0)
print(tf.keras.utils.to_categorical(lab).shape)
notrest=np.where(lab==1)
sub1, emg1, lab1 = (x[rest] for x in [sub, emg, lab])
sub2, emg2, lab2 = (x[notrest] for x in [sub, emg, lab])
fig = plt.figure(figsize=(20,20))
for i in range(np.unique(sub1).shape[0]):
    ax = fig.add_subplot(np.unique(sub1).shape[0],2, i+1)
    #c = cm.Paired(i/len(imps.keys()), 1)
    ax.plot(emg[np.where(sub1==np.unique(sub1)[i])])
    ax.set_title(i)
fig.tight_layout()
plt.show()


#print(lab)

#plt.plot(train.emg[train.act][0, :, :])
#plt.show()
#
#test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
#       None, validation=True, by_subject = True, batch_size=batch, scale = True)


#original_dim = train[0][0].shape[-1]
#
#input_shape = (original_dim, )
#intermediate_dim = 100
#batch_size = 512
#latent_dim = 2
#epochs = 50
#
#inputs = Input(shape=input_shape, name='encoder_input')
#x = Dense(intermediate_dim, activation='relu')(inputs)
#x = Dense(1000, activation='relu')(x)
#z_mean = Dense(latent_dim, name='z_mean')(x)
#z_log_var = Dense(latent_dim, name='z_log_var')(x)
#
#
#def sampling(args):
#    """Reparameterization trick by sampling from an isotropic unit Gaussian.
#    # Arguments
#        args (tensor): mean and log of variance of Q(z|X)
#    # Returns
#        z (tensor): sampled latent vector
#    """
#
#    z_mean, z_log_var = args
#    batch = K.shape(z_mean)[0]
#    dim = K.int_shape(z_mean)[1]
#    # by default, random_normal has mean = 0 and std = 1.0
#    epsilon = K.random_normal(shape=(batch, dim))
#    return(z_mean + K.exp(0.5 * z_log_var) * epsilon)
#
#
#
#z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#
#decoder_h = Dense(1000, activation='relu')(z)
#h_decoded = Dense(intermediate_dim, activation='relu')(decoder_h)
#decoder_mean = Dense(original_dim, activation='sigmoid')
#
#x_decoded_mean = decoder_mean(h_decoded)
#
## instantiate encoder model
#encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()
#
#latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#x = Dense(intermediate_dim, activation='relu')(latent_inputs)
#outputs = Dense(original_dim, activation='sigmoid')(x)
## instantiate decoder model
#decoder = Model(latent_inputs, outputs, name='decoder')
#decoder.summary()
#
#outputs = decoder(encoder(inputs)[2])
#vae = Model(inputs, outputs, name='vae_mlp')
#vae.summary()
#
#from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy
#reconstruction_loss = binary_crossentropy(inputs,outputs)
#reconstruction_loss *= original_dim
#kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#kl_loss = K.sum(kl_loss, axis=-1)
#kl_loss *= -0.5
#vae_loss = K.mean((reconstruction_loss + kl_loss))
#vae.add_loss(K.abs(vae_loss))
#
#vae.compile(optimizer=Adam(clipnorm = 1.))
#vae.fit(train,
#                epochs=50,
#                validation_data=test, shuffle = False, workers=8, use_multiprocessing=True)
#
#plt.style.use('fivethirtyeight')
#def plot_latent(model,
#                 data,
#                 batch_size=512,
#                 model_name="vae_mnist"):
#    """Plots labels and MNIST digits as a function of the 2D latent vector
#    # Arguments
#        models (tuple): encoder and decoder models
#        data (tuple): test data and label
#        batch_size (int): prediction batch size
#        model_name (string): which model is using this function
#    """
#
#    encoder = model
#    x_test, y_test = data
#
#
#    # display a 2D plot of the digit classes in the latent space
#    z_mean, _, _ = encoder.predict(x_test
#                                   )
#    plt.figure(figsize=(12, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=(y_test))
#    plt.colorbar()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.show()
#
#plot_latent(encoder, test)
