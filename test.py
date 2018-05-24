import torch
import pickle
import pandas as pd



with open("data/processed/synthetic", "rb") as f:
    (X_train, y_train, X_test, y_test) = pickle.load(f)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
