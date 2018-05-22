import numpy as np
import tensorflow as tf
import pandas as pd
import math
import tflearn
import os
import matplotlib.pyplot as plt
from random import randint
from sklearn import model_selection
from time import time
from scipy.stats import norm

from DSEBM import *
from DSEBM_parsing import *
from LSTM_AD import *
from evaluation import *
from synthetic_data_generator import *
from src.algorithms.dagmm import DAGMM
from src.datasets.dataset import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore


NUM_EPOCHS=100
NUM_EXAMPLES=100

def main():
    #y_test = binary label (1=outlier, 0=normal)
    file_name = "./data/kddcup.data_10_percent"
    
    #print("Load Data...")
    #X_train, Xy_full, y_test = load(file_name)
    #num_params = 49
    #print(X_train.shape)
    #print("Loaded data")
    
    print("Load Synthetic Data")
    df1, df2, df3, df4 = generate_outliers()
    
    y_ = []
    
    dagmm = DAGMM()
    kdd_cup = KDD_Cup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print("Trained DAGMM")
    print(get_accuracy_precision_recall_fscore(y_test, pred))
    
    
    #print("Train models...")
    #DSEBM_model = FC_DSEBM([num_params,5], num_epochs=NUM_EPOCHS)
    #DSEBM_model.fit(X_train)
    #y_.append([DSEBM_model.encode.eval(session = DSEBM_model.tf_session)])
    #print("Trained DSEBM")
    
    #len_sequence = 50
    #LSTM_AD_y_ = train(Xy_full, len_sequence)
    #print("Trained LSTM_AD")
    #y_.append(LSTM_AD_y_)
    
    evaluate(y, y_, names)

if __name__ == '__main__':
    main()