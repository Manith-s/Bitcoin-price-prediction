"""Utility functions for the Bitcoin price prediction project"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MODEL_DIR, RESULTS_DIR, LOG_DIR

# Setup logging
def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default logger
logger = setup_logger('bitcoin_prediction', 'btc_prediction.log')

def save_model(model, filename):
    """Save a model to disk.
    
    Args:
        model: The model to save
        filename: The filename without extension
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    filepath = os.path.join(MODEL_DIR, f"{filename}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")

def load_model(filename):
    """Load a model from disk.
    
    Args:
        filename: The filename without extension
        
    Returns:
        The loaded model
    """
    filepath = os.path.join(MODEL_DIR, f"{filename}.pkl")
    
    if not os.path.exists(filepath):
        logger.error(f"Model file not found: {filepath}")
        return None
        
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model

def save_dataframe(df, filepath):
    """Save a dataframe to CSV.
    
    Args:
        df: The dataframe to save
        filepath: The full path to the CSV file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    df.to_csv(filepath)
    logger.info(f"DataFrame saved to {filepath}")

def load_dataframe(filepath):
    """Load a dataframe from CSV.
    
    Args:
        filepath: The full path to the CSV file
        
    Returns:
        The loaded dataframe
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
        
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info(f"DataFrame loaded from {filepath}")
    return df

def save_results(results, filename):
    """Save results to JSON.
    
    Args:
        results: The results to save
        filename: The filename without extension
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    filepath = os.path.join(RESULTS_DIR, f"{filename}.json")
    
    # Convert numpy values to Python types for JSON serialization
    results_copy = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    
    with open(filepath, "w") as f:
        json.dump(results_copy, f, indent=4)
    logger.info(f"Results saved to {filepath}")

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics.
    
    Args:
        y_true: The true values
        y_pred: The predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays if they are not
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # For directional accuracy (classification metrics)
    y_true_direction = np.sign(np.diff(np.append([0], y_true)))
    y_pred_direction = np.sign(np.diff(np.append([0], y_pred)))
    
    # Avoiding division by zero in metrics
    try:
        acc = accuracy_score(y_true_direction, y_pred_direction)
        precision = precision_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
        recall = recall_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
        f1 = f1_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0)
    except Exception as e:
        logger.warning(f"Error calculating classification metrics: {str(e)}")
        acc, precision, recall, f1 = 0, 0, 0, 0
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    return metrics

def plot_price_prediction(actual, predicted, title="Bitcoin Price Prediction", save_path=None):
    """Plot actual vs predicted prices.
    
    Args:
        actual: The actual prices
        predicted: The predicted prices
        title: The plot title
        save_path: The path to save the plot
    """
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(predicted, label='Predicted Prices', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title="Directional Prediction Confusion Matrix", save_path=None):
    """Plot confusion matrix for directional predictions.
    
    Args:
        y_true: The true values
        y_pred: The predicted values
        title: The plot title
        save_path: The path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Convert to numpy arrays if they are not
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true_direction = np.sign(np.diff(np.append([0], y_true)))
    y_pred_direction = np.sign(np.diff(np.append([0], y_pred)))
    
    cm = confusion_matrix(y_true_direction, y_pred_direction)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def plot_feature_importance(model, feature_names, model_name, top_n=20, save_path=None):
    """Plot feature importance for a model.
    
    Args:
        model: The trained model
        feature_names: List of feature names
        model_name: The name of the model
        top_n: Number of top features to show
        save_path: The path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} doesn't have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:min(top_n, len(indices))]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {len(top_features)} Features for {model_name}')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()