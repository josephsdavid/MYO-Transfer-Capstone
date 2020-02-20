import wavenet as w
import utils as u
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import callbacks as cb


batch = 3000

class oneHotNina(u.NinaGenerator):
    def __init__(self, path: str, excercises: list,
            process_fns: list,
            augment_fns: list,
            scale=False,
            rectify=False,
            step =5, window_size=52,
            batch_size=400, shuffle=True,
            validation=False,
            by_subject=False,
            sample_0=True):
        super().__init__(
            path,
            excercises,
            process_fns,
            augment_fns,
            scale,
            rectify,
            step ,
            window_size,
            batch_size,
            shuffle,
            validation,
            by_subject,
            sample_0
        )
        self.labels = tf.keras.utils.to_categorical(self.labels)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.emg[indexes,:,:]
        if self.augmentors is not None:
            for f in self.augmentors:
                for i in range(out.shape[0]):
                    out[i,:,:]=f(out[i,:,:])
        if self.rectify:
            out = np.abs(out)
        if self.scale:
            out = scale(out)

        return out,  self.labels[indexes, :]



abc = ['b','a','c']
subject=[True, False]

results = {}
hl = []
strategy = tf.distribute.MirroredStrategy()

for i in range(len(abc)):
    for s in subject:
        train = oneHotNina("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                                [u.add_noise_snr], validation=False, by_subject = True, batch_size=batch,
                                scale = False, rectify=False, sample_0=False)
        test = oneHotNina("../data/ninaPro", [abc[i]], [u.butter_highpass_filter],
                               None, validation=True, by_subject = s, batch_size=batch,
                               scale = False, rectify = False, sample_0=False)



        x_shape = train[0][0].shape[1:]
        y_shape = (train[0][1].shape[-1], )

        #import pdb; pdb.set_trace()  # XXX BREAKPOINT
        clr=cb.OneCycleLR(
            max_lr=3,
            end_percentage=0.1,
            scale_percentage=None,
            maximum_momentum=0.95,
            minimum_momentum=0.85,
            verbose=True)
        with strategy.scope():
            wn = w.WaveNet(x_shape, y_shape).get_model()
            wn.summary()
            wn.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=['accuracy'])


        sub = 'subject' if s else 'repetition'
        loc = 'wavenet_'+sub+'_'+abc[i]
        print('beginning ' +loc )
        weight_dict = {}
        for i in range(y_shape[0]):
            if i == 0:
                w = 1/(2*y_shape[0])
            else:
                w = 1
            weight_dict[i] = w
        print(weight_dict)
        wn.fit(train, validation_data=test, epochs=500, callbacks=[clr, ModelCheckpoint('result/{}.h5'.format(loc),monitor='val_loss', save_best_only=True)], shuffle = False, class_weight=weight_dict)
        wn.save()
