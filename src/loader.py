import tensorflow as tf
import numpy as np
import utils as u
import callbacks as cb
import kerastuner as kt
from tensorflow.keras.models import load_model



def holder(a,b):
	return a, b

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise], batch_size = 3000, scale=False
                                )
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [holder], batch_size = 3000, scale=False)

lstm = load_model("result/fancy_training.h5")
lstm.evaluate(val_set)
