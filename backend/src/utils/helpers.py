"""
Helper utilities for financial indices prediction
Author: rajshash

This module provides helper functions for data processing, model evaluation,
and visualization of financial data and model results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_directory_structure(base_dir):
    """Create a directory structure for the project.
    
    Args:
        base_dir (str): Base directory path.
        
    Returns:
        dict: Dictionary with paths to created directories.
    """
    # Define directory paths
    dirs = {
        'data': os.path.join(base_dir, 'data'),
        'raw': os.path.join(base_dir, 'data', 'raw'),
        'processed': os.path.join(base_dir, 'data', 'processed'),
        'features': os.path.join(base_dir, 'data', 'features'),
        'models': os.path.join(base_dir, 'models'),
        'checkpoints': os.path.join(base_dir, 'models', 'checkpoints'),
        'results': os.path.join(base_dir, 'results'),
        'visualizations': os.path.join(base_dir, 'results', 'visualizations')
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def normalize_data(data, method='standard', return_scaler=False):
    """Normalize data using different methods.
    
    Args:
        data (numpy.ndarray): Data to normalize.
        method (str): Normalization method ('standard' or 'minmax').
        return_scaler (bool): Whether to return the scaler object.
        
    Returns:
        tuple or numpy.ndarray: Normalized data and scaler if return_scaler=True,
            otherwise just the normalized data.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Handle 1D arrays
    if len(data.shape) == 1:
        data_reshaped = data.reshape(-1, 1)
        normalized_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        normalized_data = scaler.fit_transform(data)
    
    if return_scaler:
        return normalized_data, scaler
    else:
        return normalized_data


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics.
    
    Args:
        y_true (numpy.ndarray or torch.Tensor): True values.
        y_pred (numpy.ndarray or torch.Tensor): Predicted values.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Convert torch tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Flatten arrays if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }
    
    return metrics


def plot_training_history(history, save_path=None):
    """Plot training history.
    
    Args:
        history (dict): Training history from model training.
        save_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    metric_keys = [key for key in history.keys() if key not in ['loss', 'val_loss']]
    if metric_keys:
        plt.subplot(1, 2, 2)
        for key in metric_keys:
            if history[key]:  # Check if there's data
                plt.plot(history[key], label=key)
        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_predictions(y_true, y_pred, title='Model Predictions vs Actual Values', 
                     xlabel='Time', ylabel='Value', save_path=None):
    """Plot model predictions against actual values.
    
    Args:
        y_true (numpy.ndarray or torch.Tensor): True values.
        y_pred (numpy.ndarray or torch.Tensor): Predicted values.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    # Convert torch tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Flatten arrays if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_feature_importance(feature_names, importances, title='Feature Importance', 
                           save_path=None):
    """Plot feature importance.
    
    Args:
        feature_names (list): List of feature names.
        importances (numpy.ndarray or torch.Tensor): Array of feature importances.
        title (str): Plot title.
        save_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    # Convert torch tensor to numpy if needed
    if isinstance(importances, torch.Tensor):
        importances = importances.cpu().numpy()
    
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.title(title)
    plt.xlabel('Importance')
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_correlation_matrix(data, feature_names=None, title='Feature Correlation Matrix',
                           save_path=None):
    """Plot correlation matrix of features.
    
    Args:
        data (numpy.ndarray, torch.Tensor, or pandas.DataFrame): Feature data.
        feature_names (list, optional): List of feature names.
        title (str): Plot title.
        save_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=feature_names)
    
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def create_sequences(data, sequence_length):
    """Create sequences for time series prediction.
    
    Args:
        data (numpy.ndarray or torch.Tensor): Input data, shape (n_samples, n_features).
        sequence_length (int): Length of each sequence.
        
    Returns:
        tuple: (X, y) where X is the sequence data and y is the target values.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    X = []
    y = []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        # For prediction, we use the next value after the sequence
        y.append(data[i + sequence_length, 0])  # Assuming target is the first feature
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert back to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    return X_tensor, y_tensor


def load_yaml_config(config_path):
    """Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    import yaml
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    return config


def create_data_loaders(X, y, batch_size=32, val_split=0.2, shuffle=True):
    """Create DataLoader objects for training and validation.
    
    Args:
        X (numpy.ndarray or torch.Tensor): Input features.
        y (numpy.ndarray or torch.Tensor): Target values.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of data to use for validation.
        shuffle (bool): Whether to shuffle the data.
        
    Returns:
        tuple: (train_loader, val_loader) or (train_loader, None) if val_split=0.
    """
    # Convert to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # Split dataset if validation split > 0
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    else:
        # Create only training data loader
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        
        return train_loader, None 