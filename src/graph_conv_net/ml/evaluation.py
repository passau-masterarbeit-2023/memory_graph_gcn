from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """
    Evaluate several metrics for the given true and predicted labels.
    
    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    - y_proba: array-like, predicted probabilities for the positive class
    
    Returns:
    - Dictionary containing the metrics
    """
    # compute imbalance ratio
    nb_positive_labels = np.sum(y_true)
    nb_negative_labels = len(y_true) - nb_positive_labels
    imbalance_ratio = nb_negative_labels / nb_positive_labels
    print("imbalance_ratio: {0}".format(imbalance_ratio))

    # classical metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
    return metrics