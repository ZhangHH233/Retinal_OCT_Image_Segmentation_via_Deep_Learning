import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(y_true, y_pred):
    """
    Calculate Accuracy.
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        accuracy_score: Accuracy score
    """
    true_positives = np.sum(y_true * y_pred)
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    total = np.prod(y_true.shape)
    accuracy_score = (true_positives + true_negatives) / total
    return accuracy_score

def sensitivity(y_true, y_pred):
    """
    Calculate Sensitivity/Recall/True Positive Rate.
    Sensitivity = TP/(TP + FN)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        sensitivity_score: Sensitivity score
    """
    true_positives = np.sum(y_true * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))
    sensitivity_score = true_positives / (true_positives + false_negatives + 1e-7)
    return sensitivity_score

def precision(y_true, y_pred):
    """
    Calculate Precision/Positive Predictive Value.
    Precision = TP/(TP + FP)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        precision_score: Precision score
    """
    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)
    precision_score = true_positives / (true_positives + false_positives + 1e-7)
    return precision_score

def specificity(y_true, y_pred):
    """
    Calculate Specificity/True Negative Rate.
    Specificity = TN/(TN + FP)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        specificity_score: Specificity score
    """
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    false_positives = np.sum((1 - y_true) * y_pred)
    specificity_score = true_negatives / (true_negatives + false_positives + 1e-7)
    return specificity_score

def auc_score(y_true, y_pred):
    """
    Calculate Area Under the ROC Curve.
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask (probability scores)
    Returns:
        auc: Area under ROC curve score
    """
    # Flatten the masks
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    try:
        auc = roc_auc_score(y_true_flat, y_pred_flat)
    except ValueError:
        # Handle cases where only one class is present
        auc = 0.0
        
    return auc
