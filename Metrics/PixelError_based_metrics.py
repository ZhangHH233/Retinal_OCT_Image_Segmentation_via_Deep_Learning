import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between two masks.
    MSE = 1/n * Σ(y_true - y_pred)^2
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        mse: Mean squared error
    """
    # Calculate squared differences
    squared_diff = (y_true.astype(float) - y_pred.astype(float)) ** 2
    
    # Calculate mean
    mse = np.mean(squared_diff)
    
    return mse

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between two masks.
    RMSE = sqrt(MSE) = sqrt(1/n * Σ(y_true - y_pred)^2)
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    Returns:
        rmse: Root mean squared error
    """
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    
    return rmse
