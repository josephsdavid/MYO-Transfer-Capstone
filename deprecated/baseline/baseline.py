import numpy as np
from sklearn.ensemble import RandomForestClassifier



x_tr = np.load("../data/x_train_small.npy")
y_tr = np.load("../data/y_train_small.npy")
x_val = np.load("../data/x_val_small.npy")
y_val = np.load("../data/y_val_small.npy")
x_flat = np.array([x_tr[i,:,:].flatten('F') for i in range(x_tr.shape[0])])
v_flat = np.array([x_val[i,:,:].flatten('F') for i in range(x_val.shape[0])])
del x_tr
del x_val

import gc
gc.collect()

rf = RandomForestClassifier(n_jobs = 8, n_estimators = 1000, verbose = 3, min_samples_leaf=500)


rf.fit(x_flat, y_tr)

from sklearn.metrics import accuracy_score
preds = rf.predict(v_flat)
print(accuracy_score(preds, y_val))

import joblib; joblib.dump(rf, "forest.obj")
