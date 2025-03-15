"""
LSTM Model for financial indices prediction
Author: rajshash

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting financial indices such as the S&P 500.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM Model for financial time series prediction.
    
    This class implements a multi-layer LSTM neural network model
    for predicting financial indices. It follows the architecture
    specified in the project documentation.
    
    Attributes:
        Inherits all attributes from BaseModel.
    """
    
    def __init__(self, config_path=None):
        """Initialize the LSTM model.
        
        Args:
            config_path (str, optional): Path to model configuration file.
        """
        super().__init__(config_path)
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.output_layer = None
        
    def build_model(self, input_shape):
        """Build the LSTM neural network model.
        
        This method implements the specific LSTM architecture for
        financial indices prediction as described in the README.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features).
                
        Returns:
            None: The model structure is set up internally.
        """
        # Get model parameters from config
        lstm_units = self.config['model']['lstm_units']
        dense_units = self.config['model']['dense_units']
        dropout_rate = self.config['model']['dropout_rate']
        
        sequence_length, n_features = input_shape
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=n_features,
                hidden_size=lstm_units[0],
                batch_first=True,
                return_sequences=True if len(lstm_units) > 1 else False
            )
        )
        
        # First dropout layer
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_units[i-1],
                    hidden_size=units,
                    batch_first=True,
                    return_sequences=return_sequences
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Dense layers
        input_size = lstm_units[-1]
        for units in dense_units:
            self.dense_layers.append(nn.Linear(input_size, units))
            self.dense_layers.append(nn.ReLU())
            input_size = units
        
        # Output layer
        self.output_layer = nn.Linear(input_size, 1)
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, n_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Pass through LSTM layers with dropout
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            # Check if we need to keep sequence for next LSTM layer
            if i < len(self.lstm_layers) - 1:
                x, _ = lstm(x)
                x = dropout(x)
            else:
                # Last LSTM layer
                if lstm.batch_first:
                    # If return_sequences=False, we get (batch_size, hidden_size)
                    if not getattr(lstm, 'return_sequences', False):
                        x, _ = lstm(x)
                        x = dropout(x)
                    else:
                        # If return_sequences=True, we get (batch_size, seq_length, hidden_size)
                        # Take the last output of the sequence
                        x, _ = lstm(x)
                        x = dropout(x)
                        x = x[:, -1, :]
                else:
                    # If not batch_first, adjust accordingly
                    x = x.permute(1, 0, 2)  # (seq_length, batch_size, n_features)
                    x, _ = lstm(x)
                    x = x[-1]  # Take the last output
                    x = dropout(x)
        
        # Pass through dense layers
        for i in range(0, len(self.dense_layers), 2):
            linear = self.dense_layers[i]
            activation = self.dense_layers[i+1]
            x = activation(linear(x))
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def prepare_data_loader(self, data, targets=None, batch_size=None, shuffle=True):
        """Prepare a DataLoader for the model.
        
        Args:
            data (numpy.ndarray): Input data of shape (n_samples, n_features) or
                (n_samples, seq_length, n_features).
            targets (numpy.ndarray, optional): Target values. If None, sequences
                will be created from the data.
            batch_size (int, optional): Batch size. If None, uses the value from the configuration.
            shuffle (bool): Whether to shuffle the data.
                
        Returns:
            torch.utils.data.DataLoader: DataLoader for the provided data.
        """
        if batch_size is None:
            batch_size = self.config['model']['batch_size']
        
        # If data is 2D and targets are not provided, create sequences
        if len(data.shape) == 2 and targets is None:
            X, y = self.prepare_sequences(data)
        elif targets is not None:
            # Convert to PyTorch tensors
            X = torch.tensor(data, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        else:
            X = torch.tensor(data, dtype=torch.float32)
            y = torch.zeros((X.shape[0], 1), dtype=torch.float32)  # Dummy targets
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        
        return dataloader
    
    def prepare_sequences(self, data, sequence_length=None):
        """Prepare sequences for LSTM model training.
        
        This method transforms a time series into sequences suitable
        for LSTM training, where each sequence is paired with the next
        value as its target.
        
        Args:
            data (numpy.ndarray): Time series data, shape (n_samples, n_features).
            sequence_length (int, optional): Length of each sequence. If None,
                uses the value from the configuration.
                
        Returns:
            tuple: (X, y) where X is the sequence data and y is the target values.
        """
        if sequence_length is None:
            sequence_length = self.config['model']['sequence_length']
        
        X = []
        y = []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            # For prediction, we use the next closing price after the sequence
            y.append(data[i + sequence_length, 0])  # Assuming closing price is the first feature
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        return X_tensor, y_tensor
    
    def predict_sequence(self, initial_sequence, steps=30):
        """Generate multi-step predictions using the model.
        
        This method makes recursive predictions for multiple steps ahead,
        using each prediction as input for the next prediction.
        
        Args:
            initial_sequence (numpy.ndarray or torch.Tensor): Initial sequence data,
                shape (1, sequence_length, n_features).
            steps (int): Number of steps to predict ahead.
                
        Returns:
            numpy.ndarray: Array of predicted values for each step.
        """
        # Make sure the model is in evaluation mode
        self.eval()
        
        sequence_length = self.config['model']['sequence_length']
        
        # Convert to PyTorch tensor if necessary
        if isinstance(initial_sequence, np.ndarray):
            current_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
        else:
            current_sequence = initial_sequence.clone()
        
        # Check if initial_sequence has the correct shape
        if current_sequence.shape[1] != sequence_length:
            raise ValueError(f"Initial sequence must have shape (1, {sequence_length}, n_features)")
        
        # Move to the same device as the model
        current_sequence = current_sequence.to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                # Make a prediction
                next_pred = self(current_sequence)
                predictions.append(next_pred.item())
                
                # Update the sequence by removing the first element and adding the new prediction
                new_point = current_sequence[0, -1, :].clone()
                new_point[0] = next_pred.item()  # Set the first feature (closing price) to the prediction
                
                # Remove first time step and add new point at the end
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    new_point.view(1, 1, -1)
                ], dim=1)
        
        return np.array(predictions) 