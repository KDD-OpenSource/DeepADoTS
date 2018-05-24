import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support as prf, accuracy_score

def get_accuracy_precision_recall_fscore(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f_score
    
def compute_roc_curve(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc[0] = auc(fpr[0], tpr[0])

    #Plot of a ROC curve for a specific class

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def evaluate(y_true, y_pred, names):
    df = pd.DataFrame(columns = ["name", "accuracy", "precision", "recall", "F1-score"])
    for i in range(len(names)):
        acc, prec, rec, f_score = get_accuracy_precision_recall_fscore(y_true, y_pred[i])
        df = df.append({"name": names[i], 
                   "accuracy": acc,
                   "precision": prec, 
                   "recall": rec, 
                   "F1-score": f_score},
                   ignore_index=True)

        compute_roc_curve(y_true, y_pred[i])
    print(df)