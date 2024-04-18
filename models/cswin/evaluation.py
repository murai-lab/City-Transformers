import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score

def onehot(all_labels, numofcat=4):
    onehot_all_labels = np.zeros((len(all_labels), numofcat))
    for i in range(len(all_labels)):
        onehot_all_labels[i][all_labels[i]-1] = 1
    return onehot_all_labels



def Eval(all_decisions, all_labels, numofcat=4):
    confusion = confusion_matrix(all_labels, all_decisions)
    micro_recall = recall_score(all_labels, all_decisions, average='micro')
    macro_recall = recall_score(all_labels, all_decisions, average='macro')
    weighted_recall = recall_score(all_labels, all_decisions, average='weighted')
    micro_precision = precision_score(all_labels, all_decisions, average='micro')
    macro_precision = precision_score(all_labels, all_decisions, average='macro')
    weighted_precision = precision_score(all_labels, all_decisions, average='weighted')
    micro_f1 = f1_score(all_labels, all_decisions, average='micro')
    macro_f1 = f1_score(all_labels, all_decisions, average='macro')
    weighted_f1 = f1_score(all_labels, all_decisions, average='weighted')
    # onehot_all_labels = onehot(all_labels, numofcat)
    # roc_auc = roc_auc_score(all_labels, all_predictions)
    # pr_auc = average_precision_score(all_labels, all_predictions)
    return {
        'confusion_matrix': confusion.tolist(),
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'weighted_recall': weighted_recall,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'weighted_precision': weighted_precision,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }