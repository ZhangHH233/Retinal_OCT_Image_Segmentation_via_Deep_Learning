import numpy as np

def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice Similarity Coefficient.
    DSC = 2|X∩Y|/(|X|+|Y|)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        dice_score: Dice Similarity Coefficient score
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice_score = (2. * intersection) / (union + 1e-7)
    return dice_score

def iou_score(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU/Jaccard Index).
    IoU = |X∩Y|/|X∪Y|
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        iou: IoU score
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = intersection / (union + 1e-7)
    return iou

def precision(y_true, y_pred):
    """
    Calculate Precision.
    Precision = TP/(TP+FP)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        precision_score: Precision score
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    precision_score = true_positives / (predicted_positives + 1e-7)
    return precision_score

def recall(y_true, y_pred):
    """
    Calculate Recall.
    Recall = TP/(TP+FN)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        recall_score: Recall score
    """
    true_positives = np.sum(y_true * y_pred)
    actual_positives = np.sum(y_true)
    recall_score = true_positives / (actual_positives + 1e-7)
    return recall_score
