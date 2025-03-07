import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

def hausdorff_distance(y_true, y_pred):
    """
    Calculate Hausdorff Distance between two binary masks.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        hd: Hausdorff distance
    """
    # Get contours
    y_true_contour = measure.find_contours(y_true, 0.5)[0]
    y_pred_contour = measure.find_contours(y_pred, 0.5)[0]
    
    # Calculate distances from each point to the other contour
    d1 = np.max([np.min(np.sqrt(np.sum((y_true_contour - p) ** 2, axis=1))) for p in y_pred_contour])
    d2 = np.max([np.min(np.sqrt(np.sum((y_pred_contour - p) ** 2, axis=1))) for p in y_true_contour])
    
    return max(d1, d2)

def hausdorff_distance_95(y_true, y_pred):
    """
    Calculate 95th percentile Hausdorff Distance.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        hd95: 95th percentile Hausdorff distance
    """
    y_true_contour = measure.find_contours(y_true, 0.5)[0]
    y_pred_contour = measure.find_contours(y_pred, 0.5)[0]
    
    d1 = [np.min(np.sqrt(np.sum((y_true_contour - p) ** 2, axis=1))) for p in y_pred_contour]
    d2 = [np.min(np.sqrt(np.sum((y_pred_contour - p) ** 2, axis=1))) for p in y_true_contour]
    
    return max(np.percentile(d1, 95), np.percentile(d2, 95))

def assd(y_true, y_pred):
    """
    Calculate Average Symmetric Surface Distance.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        assd_score: Average symmetric surface distance
    """
    y_true_contour = measure.find_contours(y_true, 0.5)[0]
    y_pred_contour = measure.find_contours(y_pred, 0.5)[0]
    
    d1 = np.mean([np.min(np.sqrt(np.sum((y_true_contour - p) ** 2, axis=1))) for p in y_pred_contour])
    d2 = np.mean([np.min(np.sqrt(np.sum((y_pred_contour - p) ** 2, axis=1))) for p in y_true_contour])
    
    return (d1 + d2) / 2

def mad(y_true, y_pred):
    """
    Calculate Mean Absolute Difference between two binary masks.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        mad_score: Mean absolute difference
    """
    # Calculate absolute difference
    abs_diff = np.abs(y_true.astype(float) - y_pred.astype(float))
    
    # Calculate mean
    mad_score = np.mean(abs_diff)
    
    return mad_score
