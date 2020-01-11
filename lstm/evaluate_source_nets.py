import numpy as np
from dataloaders import train_loader, val_loader, test_loader

X_train, y_train = train_loader("../EvaluationDataset")

X_val, y_val = val_loader("../EvaluationDataset")

X_test, y_test = test_loader("../EvaluationDataset")


