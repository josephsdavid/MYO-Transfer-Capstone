import numpy as np
from dataloaders import train_loader, val_loader, test_loader
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import load_model

X_train, y_train = train_loader("../EvaluationDataset")

X_val, y_val = val_loader("../EvaluationDataset")

X_test, y_test = test_loader("../EvaluationDataset")
y_test = to_categorical(y_test)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


score_simp = load_model("models/simple_lstm.h5").evaluate(X_test, y_test)




score_wide = load_model("models/wide_lstm.h5").evaluate(X_test, y_test)
