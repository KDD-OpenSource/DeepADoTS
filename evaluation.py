import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pandas_ml import ConfusionMatrix

def evaluate_seq(y, saver_locations, names):
    saver = tf.train.Saver()
    
    df = pd.Dataframe(columns = ["name", "precision", "recall", "F1-score"])
    for i in range(len(names)):
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, saver_locations[i])
            print("Model restored.")

            y_ = y_.eval()

            cm = ConfusionMatrix(y, y_)
            cm.print_stats()
            df.append({"name": names[i], 
                       "precision": curr_cm['PPV'], 
                       "recall": curr_cm['TPR'], 
                       "F1-score": curr_cm['F1']})
            
            compute_roc_curve()
    print(df)
    
def evaluate(y, y_, names):
    saver = tf.train.Saver()
    
    df = pd.Dataframe(columns = ["name", "precision", "recall", "F1-score"])
    for i in range(len(names)):

        cm = ConfusionMatrix(y, y_[i])
        cm.print_stats()
        df.append({"name": names[i], 
                   "precision": curr_cm['PPV'], 
                   "recall": curr_cm['TPR'], 
                   "F1-score": curr_cm['F1']})

        compute_roc_curve()
    print(df)
    
def compute_roc_curve(y, y_):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y, y_, pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

        #Plot of a ROC curve for a specific class

        plt.figure()
        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
def DBSM_evaluate():
    newdata = []
    newresult = []
    outliers = 0
    for i in range(NUM_EXAMPLES):
        newindex = randint(0,len(data)-1)
        newdata += [Xy_full[newindex]]
        newresult += [y_test[newindex]]
        if newresult[i] == 1:
            outliers += 1

    
    score = model.score(newdata, newresult, 10e+4)
    print("Score: {} (ouliers ratio: {})".format(score, outliers/NUM_EXAMPLES))
    model.delete()