import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    
    Returns:
    float: MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_metrics(y_true, y_pred):
    """
    Evaluate multiple error metrics for the model predictions.
    
    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    
    Returns:
    dict: Dictionary containing different error metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'Mean Squared Error (MSE)': mse,
        'Mean Absolute Error (MAE)': mae,
        'Mean Absolute Percentage Error (MAPE)': mape
    }
