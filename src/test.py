import tensorflow as tf
import numpy as np
import utils as u

train_set = u.PreTrainGenerator("../EvaluationDataset", [u.butter_highpass_filter], [u.add_noise])
val_set = u.PreValGenerator("../PreTrainingDataset", [u.butter_highpass_filter], [u.add_noise])

print(len(train_set))
print(train_set.__getitem__(0)[0].shape)
print(len(val_set))
print(val_set.__getitem__(0)[0].shape)
