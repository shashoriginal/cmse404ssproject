"""
Training utilities for financial prediction models
Author: rajshash

This module provides utilities for training, evaluating, and running experiments
with financial prediction models.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import yaml

from .lstm import LSTMModel


def load_and_prepare_data(data_path, sequence_length=60, train_test_split=0.8, standardize=True, batch_size=None):
    """Load and prepare data for model training.
    
    Args:
        data_path (str): Path to the processed data file.
        sequence_length (int): Length of input sequences.
        train_test_split (float): Ratio of training data to total data.
        standardize (bool): Whether to standardize the features.
        batch_size (int, optional): Batch size for DataLoader. If None, use default from model config.
        
    Returns:
        tuple: (train_loader, test_loader, scaler)
    """
    # Load the data
    data = np.load(data_path)
    
    # Standardize the data
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        scaler = None
    
    # Create sequences
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        # For prediction, we use the next closing price after the sequence
        y.append(data[i + sequence_length, 0])  # Assuming closing price is the first feature
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    split_idx = int(len(X) * train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    batch_size = batch_size or 32  # Default batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


def train_lstm_model(train_loader, val_loader=None, config_path=None, model_save_path=None, num_epochs=None):
    """Train an LSTM model for financial prediction.
    
    Args:
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
        config_path (str, optional): Path to the model configuration file.
        model_save_path (str, optional): Path to save the trained model.
        num_epochs (int, optional): Number of epochs to train for.
        
    Returns:
        tuple: (model, history)
    """
    # Get input shape from first batch
    for inputs, _ in train_loader:
        input_shape = (inputs.shape[1], inputs.shape[2])
        break
        
    # Create the model
    model = LSTMModel(config_path)
    
    # Build the model
    model.build_model(input_shape)
    
    # Train the model
    history = model.train_model(train_loader, val_loader, num_epochs)
    
    # Save the model if a path is provided
    if model_save_path:
        model.save(model_save_path)
    
    return model, history


def evaluate_and_visualize(model, test_loader, scaler=None, save_dir=None):
    """Evaluate the model and visualize the results.
    
    Args:
        model: Trained model instance.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        scaler (sklearn.preprocessing.StandardScaler, optional): 
            Scaler used to standardize the data.
        save_dir (str, optional): Directory to save the visualizations.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Evaluate the model
    metrics = model.evaluate(test_loader)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Collect all test data and predictions for visualization
    model.eval()
    device = model.device
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate all batches
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape predictions and targets for inverse transform
        pred_reshaped = np.zeros((len(predictions), scaler.n_features_in_))
        pred_reshaped[:, 0] = predictions.flatten()
        
        y_reshaped = np.zeros((len(targets), scaler.n_features_in_))
        y_reshaped[:, 0] = targets.flatten()
        
        # Inverse transform
        pred_original = scaler.inverse_transform(pred_reshaped)[:, 0]
        y_original = scaler.inverse_transform(y_reshaped)[:, 0]
    else:
        pred_original = predictions.flatten()
        y_original = targets.flatten()
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_original, label='Actual')
    plt.plot(pred_original, label='Predicted')
    plt.title('Financial Index Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save the visualization if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'prediction_plot_{timestamp}.png'))
    
    plt.show()
    
    return metrics


def run_experiment(data_path, config_path=None, output_dir=None):
    """Run a complete experiment from data loading to evaluation.
    
    This function orchestrates the entire model training and evaluation
    process, saving results and visualizations.
    
    Args:
        data_path (str): Path to the processed data file.
        config_path (str, optional): Path to the model configuration file.
        output_dir (str, optional): Directory to save outputs.
        
    Returns:
        dict: Results of the experiment.
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, 'models')
        visuals_dir = os.path.join(output_dir, 'visualizations')
        results_dir = os.path.join(output_dir, 'results')
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    else:
        models_dir = None
        visuals_dir = None
        results_dir = None
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        # Use default config from LSTMModel
        model = LSTMModel()
        config = model.config
    
    # Extract parameters from config
    sequence_length = config['model']['sequence_length']
    batch_size = config['model']['batch_size']
    
    # Load and prepare the data
    train_loader, test_loader, scaler = load_and_prepare_data(
        data_path, 
        sequence_length=sequence_length,
        train_test_split=config['data']['preprocessing']['train_test_split'],
        batch_size=batch_size
    )
    
    # Create validation set from training data
    val_split = config['training']['validation_split']
    if val_split > 0:
        # Get total size of training data
        train_size = len(train_loader.dataset)
        val_size = int(train_size * val_split)
        train_size = train_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_loader.dataset, [train_size, val_size]
        )
        
        # Create new data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None
    
    # Set model save path if output directory is provided
    if models_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(models_dir, f'lstm_model_{timestamp}.pt')
    else:
        model_save_path = None
    
    # Train the model
    model, history = train_lstm_model(
        train_loader, val_loader, 
        config_path=config_path, 
        model_save_path=model_save_path
    )
    
    # Evaluate and visualize
    metrics = evaluate_and_visualize(model, test_loader, scaler, visuals_dir)
    
    # Save results if output directory is provided
    if results_dir:
        # Save training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(results_dir, f'history_{timestamp}.pkl')
        with open(history_path, 'wb') as file:
            pickle.dump(history, file)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, f'metrics_{timestamp}.yml')
        with open(metrics_path, 'w') as file:
            yaml.dump(metrics, file)
    
    # Compile results
    results = {
        'metrics': metrics,
        'history': history,
        'model_path': model_save_path,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return results


def main():
    """Main function to run the model training script.
    
    This function can be executed directly to train a model with default
    parameters or from command line with arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train financial prediction models')
    parser.add_argument('--data', type=str, required=True, help='Path to the processed data file')
    parser.add_argument('--config', type=str, help='Path to the model configuration file')
    parser.add_argument('--output', type=str, help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Run the experiment
    results = run_experiment(args.data, args.config, args.output)
    
    print("Experiment completed successfully!")
    print(f"Results saved to: {args.output}" if args.output else "Results not saved (no output directory specified)")


if __name__ == "__main__":
    main() 