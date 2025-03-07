import numpy as np

def thickness_difference(y_true, y_pred):
    """
    Calculate Thickness Difference (TD) between ground truth and predicted masks.
    TD measures the absolute difference in thickness between two segmentations.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        td: Mean absolute thickness difference
    """
    # Calculate thickness for each mask by counting non-zero pixels in each column
    y_true_thickness = np.sum(y_true, axis=0)
    y_pred_thickness = np.sum(y_pred, axis=0)
    
    # Calculate absolute difference in thickness
    thickness_diff = np.abs(y_true_thickness - y_pred_thickness)
    
    # Return mean thickness difference
    return np.mean(thickness_diff)

def vascularity_index(y_true, y_pred):
    """
    Calculate Vascularity Index (VI) between ground truth and predicted masks.
    VI measures the ratio of vascular pixels to total tissue pixels.
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
    Returns:
        vi_diff: Absolute difference between true and predicted vascularity indices
    """
    # Calculate VI for each mask
    vi_true = np.sum(y_true) / y_true.size
    vi_pred = np.sum(y_pred) / y_pred.size
    
    # Return absolute difference between indices
    return np.abs(vi_true - vi_pred)
