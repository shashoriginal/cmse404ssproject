"""
Base Model class for financial indices prediction
Author: rajshash

This module provides a base class for all neural network models in the project.
It implements common functionality used across different model architectures.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Base class for all neural network models.
    
    This abstract class defines the common interface and functionality
    for all neural network models in the project. It handles configuration
    loading, model setup, training, evaluation, and prediction.
    
    Attributes:
        config_path (str): Path to the model configuration file.
        config (dict): Model configuration parameters.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    
    def __init__(self, config_path=None):
        """Initialize the base model with configuration.
        
        Args:
            config_path (str, optional): Path to the model configuration file.
                If None, the default configuration will be used.
        """
        super(BaseModel, self).__init__()
        
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'configs', 'model_config.yaml'
        )
        self.config = self._load_config()
        
        # Set device (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
    def _load_config(self):
        """Load model configuration from YAML file.
        
        Returns:
            dict: Configuration parameters.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    @abstractmethod
    def build_model(self, input_shape):
        """Build the neural network model.
        
        This method must be implemented by each subclass to define
        the specific model architecture.
        
        Args:
            input_shape (tuple): Shape of the input data.
            
        Returns:
            None: The model is set up internally as layers of the nn.Module.
        """
        pass
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.
        
        This method must be implemented by each subclass to define
        how data flows through the model.
        
        Args:
            x (torch.Tensor): Input data.
            
        Returns:
            torch.Tensor: Model output.
        """
        pass
    
    def compile_model(self):
        """Set up the optimizer and loss function.
        
        Returns:
            tuple: (optimizer, criterion)
        """
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        loss_name = self.config['training']['loss']
        
        # Set up optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set up loss function
        if loss_name.lower() == 'mse':
            criterion = nn.MSELoss()
        elif loss_name.lower() == 'mae':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        return optimizer, criterion
    
    def train_model(self, train_loader, val_loader=None, num_epochs=None):
        """Train the model on the provided data.
        
        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
            num_epochs (int, optional): Number of epochs to train for.
                If None, use the value from the configuration.
                
        Returns:
            dict: Training history.
        """
        # Move model to the appropriate device
        self.to(self.device)
        
        # Set number of epochs
        if num_epochs is None:
            num_epochs = self.config['model']['epochs']
        
        # Get optimizer and loss function
        optimizer, criterion = self.compile_model()
        
        # Set up early stopping if configured
        early_stopping = False
        patience = 0
        best_val_loss = float('inf')
        
        if 'early_stopping' in self.config['training']:
            early_stopping = True
            patience = self.config['training']['early_stopping']['patience']
            wait = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.train()  # Set model to training mode
            train_loss = 0.0
            train_mae = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss and metrics
                train_loss += loss.item() * inputs.size(0)
                train_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_mae = train_mae / len(train_loader.dataset)
            
            # Validation phase
            val_loss = 0.0
            val_mae = 0.0
            
            if val_loader:
                self.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Forward pass
                        outputs = self(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Track loss and metrics
                        val_loss += loss.item() * inputs.size(0)
                        val_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                val_mae = val_mae / len(val_loader.dataset)
                
                # Check for early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        # Save the best model if checkpoint is configured
                        if 'model_checkpoint' in self.config['training']:
                            if self.config['training']['model_checkpoint']['save_best_only']:
                                checkpoint_dir = os.path.dirname(self.config['training']['model_checkpoint']['filepath'])
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                filepath = os.path.join(checkpoint_dir, f'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.pt')
                                self.save(filepath)
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f'Early stopping at epoch {epoch}')
                            break
            
            # Log progress
            if val_loader:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, MAE: {train_mae:.4f}')
            
            # Update history
            self.history['loss'].append(train_loss)
            self.history['mae'].append(train_mae)
            if val_loader:
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data.
        
        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader.
            
        Returns:
            dict: Evaluation metrics.
        """
        # Move model to the appropriate device
        self.to(self.device)
        self.eval()  # Set model to evaluation mode
        
        # Get loss function
        _, criterion = self.compile_model()
        
        # Initialize metrics
        test_loss = 0.0
        all_targets = []
        all_predictions = []
        
        # Evaluation loop
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Track loss
                test_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets for additional metrics
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        # Calculate average loss
        test_loss = test_loss / len(test_loader.dataset)
        
        # Combine all predictions and targets
        targets = np.concatenate(all_targets)
        predictions = np.concatenate(all_predictions)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(targets - predictions))
        mse = np.mean(np.square(targets - predictions))
        rmse = np.sqrt(mse)
        
        # Calculate R² score
        ss_total = np.sum(np.square(targets - np.mean(targets)))
        ss_residual = np.sum(np.square(targets - predictions))
        r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        metrics = {
            'loss': test_loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2_score
        }
        
        return metrics
    
    def predict(self, inputs):
        """Make predictions using the trained model.
        
        Args:
            inputs (numpy.ndarray or torch.Tensor): Input features.
            
        Returns:
            numpy.ndarray: Predicted values.
        """
        # Move model to the appropriate device
        self.to(self.device)
        self.eval()  # Set model to evaluation mode
        
        # Convert inputs to PyTorch tensor if necessary
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
        
        # Move inputs to the same device as the model
        inputs = inputs.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self(inputs)
        
        # Return as numpy array
        return outputs.cpu().numpy()
    
    def save(self, filepath):
        """Save the model to a file.
        
        Args:
            filepath (str): Path to save the model.
            
        Returns:
            None
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
    
    def load(self, filepath):
        """Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from.
            
        Returns:
            None
        """
        # Load the model
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.history = checkpoint.get('history', {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}) 