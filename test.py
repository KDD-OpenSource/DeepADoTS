import torch
import pickle
import pandas as pd



with open("data/mypickle", "rb") as f:
    (X_train, y_train, X_test, y_test) = pickle.load(f)

print(isinstance(X_train, pd.DataFrame))

print(torch.FloatTensor(X_train.values))
