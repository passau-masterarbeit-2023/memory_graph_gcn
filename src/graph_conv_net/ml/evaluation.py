from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Any, Dict, cast
from graph_conv_net.params.params import ProgramParams
from graph_conv_net.utils.utils import str2enum
import numpy as np
import json
import logging

from ..results.base_result_writer import BaseResultWriter, SaveFileFormat

def __get_predicted_classes_from_report(clf_report: dict) -> list:
    """
    Return the classes from the classification report.
    """
    # Create a list to hold the classes
    classes = []
    # Iterate over the keys in the classification report
    for key in clf_report.keys():
        # Ignore the 'accuracy', 'macro avg' and 'weighted avg' keys,
        # as these are not classes
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            # Append the class (as an integer) to the classes list
            classes.append(key)
    # Return the classes list
    return classes

def evaluate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        result_saver: BaseResultWriter,
        params: ProgramParams,
    ):
    """
    Evaluate several metrics for the given true and predicted labels.
    
    Parameters:
    - :y_true: array-like, true labels
    - :y_pred: array-like, predicted labels

    Returns:
    - Dictionary containing the metrics
    """
    logger : logging.Logger = params.RESULTS_LOGGER
    metrics: dict[str, int | float | str] = {}

    # compute imbalance ratio
    nb_positive_labels = np.sum(y_true)
    nb_negative_labels = len(y_true) - nb_positive_labels
    imbalance_ratio = nb_negative_labels / nb_positive_labels
    metrics["imbalance_ratio"] = imbalance_ratio

    # Get the classification report in a dictionary format
    # NOTE : cast is used to tell mypy that the return type of classification_report is a dict, not a string
    #       (no runtime effect)
    clf_report  = cast(Dict[str, Any], classification_report(y_true, y_pred, output_dict=True))

    for predicted_class in __get_predicted_classes_from_report(clf_report):
        precision_field = "precision_class_" + str(predicted_class)
        metrics[precision_field] = str(clf_report[str(predicted_class)]['precision'])
        recall_field = "recall_class_" + str(predicted_class)
        metrics[recall_field] = str(clf_report[str(predicted_class)]['recall'])
        f1_score_field = "f1_score_class_" + str(predicted_class)
        metrics[f1_score_field] = str(clf_report[str(predicted_class)]['f1-score'])
        support_field = "support_class_" + str(predicted_class)
        metrics[support_field] = str(clf_report[str(predicted_class)]['support'])

    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix: ")
    logger.info("true_positives: %s" % str(cm[1, 1]))
    logger.info("true_negatives: %s" % str(cm[0, 0]))
    logger.info("false_positives: %s" % str(cm[0, 1]))
    logger.info("false_negatives: %s" % str(cm[1, 0]))
    metrics["true_positives"] = str(cm[1, 1])
    metrics["true_negatives"] = str(cm[0, 0])
    metrics["false_positives"] = str(cm[0, 1])
    metrics["false_negatives"] = str(cm[1, 0])

    # calculate the false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

    # calculate the area under the ROC curve
    roc_auc = auc(fpr, tpr)
    metrics["AUC"] = float(roc_auc)
    logger.info("AUC: %s" % str(roc_auc))

    logger.info(json.dumps(metrics, indent=4))

    # save results
    for metric_name, metric_value in metrics.items():
        result_saver.set_result(metric_name, str(metric_value))
    result_saver.save_results_to_file(
        str2enum(params.RESULT_SAVE_FILE_FORMAT, SaveFileFormat)
    )
        
    return metrics
