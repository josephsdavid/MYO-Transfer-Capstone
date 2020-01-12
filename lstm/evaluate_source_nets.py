import numpy as np
from dataloaders import train_loader, val_loader, test_loader
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import load_model

X_train, y_train = train_loader("../EvaluationDataset", scale = True, filter = False, noise = False)

X_val, y_val = val_loader("../EvaluationDataset", scale = True, filter = False, noise = False)

X_test, y_test = test_loader("../EvaluationDataset", scale = True, filter = False, noise = False)
y_test = to_categorical(y_test)



score_simp = load_model("models/simple_lstm.h5").evaluate(X_test, y_test)




score_wide = load_model("models/wide_lstm.h5").evaluate(X_test, y_test)
