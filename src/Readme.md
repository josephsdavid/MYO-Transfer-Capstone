# Source code

We will walk through the purpose of the code in this directory, and how to use it:

* activations
	*  Activation functions are stored here:
		* Mish
		* Sparsemax activation (doesnt work)
		* Sparsemax loss
* callbacks
	* Custom Callbacks are stored here:
		* Cyclic Learning Rate
* layers
	* custom layers are stored here:
		* Temporal attention
* optimizers
	* custom optimizers are stored here:
		* Ranger
		* Lookahead
		* NovoGrad
		* RectifiedAdam
		* Yogi
* utils
	* utilities are stored here:
		* NinaGenerator
		* NinaGeneratorConv
		* NinaMA
		* butter_highpass_filter
		* scale
		* add_noise
		* add_noise_snr
		* add_noise_random
* Builders:
	* Builders for models are stored here
		* Attention based models
		* Convolutional models

* result
	* results of various tunings and slurm output her
* history
	* history objects and cleaned up slurm output, as well as history plots (please)
* old
	* old stuff, useful as a reference 

The files in this directory are as follows:
* att_tuner.py
	* keras tuner hyperband search optimizing ranger for the attention model
* builder_funs.py
	* Functions to build models used (DEPRECATED)
* conv-rnn.py
	* simple conv-rnn model
* conv.py
	* simple cnn model
* convlstm.py
	* ConvLSTM2D model
* eda.py
	* simple eda scratch work
* experiment.py
	* Current home for running the attention model
* test.py
	* nothing useful
* tuner.py 
	* Big LSTM hyperband tuner
* loss.sh
	* with proper print functions (displayed below), cleans up the slurm output into a csv
* q.sh
	* watch the slum queue (better to just use `watch -n 1 squeue -whatever`) 
* observer.sh
	* watch slurm go! usage: `/observer.sh result/slurm-324321 458754745745`



## How to test a model (old tests were deleted!)

Write a script that looks something like this:

```python
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import utils as u
import callbacks as cb
import builders.attentional as b


def make_data(exercise, by_subject):
	train = u.NinaMA("../data/ninaPro", [exercise], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=False, by_subject = by_subject, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

	test = u.NinaMA("../data/ninaPro", [exercise], [u.butter_highpass_filter],
                        [u.add_noise_random], validation=True, by_subject = by_subject, batch_size=batch,
                        scale = False, rectify=True, sample_0=False, step=5, n=15, window_size=52, super_augment=False)

	subject = 'subject' if by_subject else 'repetition'
	path_string = "{}_{}".format(subject, exercise)
	return train, test, path_string

def plot_history(history, save_path):
	plt.subplot(212)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	# summarize history for loss
	plt.subplot(211)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	F = plt.gcf()
	Size = F.get_size_inches()
	F.set_size_inches(Size[0]*2, Size[1]*2)
	plt.savefig("{}.png".format(save_path))

exercises = ['a','b','c']
subjects = [False, True]
batch = 512

for ex in exercises:
	for sub in subjects:
		train, test, string = make_data(ex, sub)
		path = "attention_lstm_{}".format(string)

		# for loss.sh
		print('beginning {}'.format(path))

		n_time = train[0][0].shape[1]
		n_class = np.unique(train.labels).shape[0]
		
		model = b.build_att_gru(n_time, n_class)
		
		class_weights = {i:1/(n_class) if i==0 else 1 for i in range(1, n_class+1)}

		history = model.fit(train, validation_data=test, epochs=100, shuflle=False,
				workers=12, use_multiprocessing=True,
				callbacks=[
					ModelCheckpoint("{}.h5".format(path), monitor="val_loss", keep_best_only=True),
					ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)],
				class_weight=class_weights)

		plot_history(history, "history/{}".format(path))

# We dont use any callbacks here but it would be like this:
#clr=cb.OneCycleLR(
#                 max_lr=.1,
#                 end_percentage=0.1,
#                 scale_percentage=None,
#                 maximum_momentum=0.95,
#                 minimum_momentum=0.85,
#                 verbose=True)
```

