from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score


def get_accuracy_precision_recall_fscore(ground_truth, prediction):
    accuracy = accuracy_score(ground_truth, prediction)
    precision, recall, f_score, support = prf(ground_truth, prediction, average='binary')
    return accuracy, precision, recall, f_score
