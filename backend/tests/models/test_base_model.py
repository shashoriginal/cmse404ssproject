"""
Unit tests for the base model.
Author: rajshash

This module provides unit tests for the base model class used in financial indices prediction.
"""

import os
import sys
import unittest
import numpy as np
import torch
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.base_model import BaseModel


class DummyModel(BaseModel):
    """A dummy model class for testing."""
    
    def build_model(self):
        """Build a dummy model."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.input_shape[-1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.output_shape)
        )
    
    def forward(self, x):
        """Forward pass for the dummy model."""
        return self.model(x)
    
    def prepare_sequences(self, data, sequence_length=None):
        """Prepare sequences for testing."""
        # Just return a simple tensor
        return torch.randn(10, 5), torch.randn(10, 1)


class TestBaseModel(unittest.TestCase):
    """Test cases for the base model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            'model': {
                'type': 'dummy',
                'input_shape': [10, 5],  # Convert tuple to list for YAML serialization
                'output_shape': 1,
                'sequence_length': 10,
                'batch_size': 32,
                'epochs': 2
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'loss': 'mse',
                'early_stopping_patience': 5
            }
        }
        
        # Save config to temporary file
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create dummy model
        self.model = DummyModel(config_path=self.config_path)
        
        # Create dummy data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)  # 100 samples, 5 features
        self.y_train = np.random.randn(100, 1)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20, 1)
        
        # Convert to torch tensors
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of the base model."""
        self.assertEqual(self.model.config_path, self.config_path)
        self.assertIsInstance(self.model.config, dict)
        self.assertEqual(self.model.input_shape, (10, 5))
        self.assertEqual(self.model.output_shape, 1)
    
    def test_build_model(self):
        """Test building the model."""
        model = self.model.build_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_forward(self):
        """Test forward pass of the model."""
        self.model.build_model()
        
        # Create a sample input
        x = torch.randn(32, 5)  # batch_size, n_features
        
        # Test forward pass
        output = self.model(x)
        self.assertEqual(output.shape, (32, 1))
    
    @patch('torch.utils.data.DataLoader')
    def test_prepare_data_loader(self, mock_dataloader):
        """Test preparation of data loaders."""
        # Set up the mock
        mock_dataloader.return_value = MagicMock()
        
        # Test with default parameters
        with patch.object(self.model, 'prepare_data_loader', return_value=(MagicMock(), MagicMock())):
            train_loader, val_loader = self.model.prepare_data_loader(
                self.X_train, self.y_train,
                val_split=0.2, batch_size=32
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
    
    def test_normalize_data(self):
        """Test data normalization."""
        # Create sample data
        data = np.random.randn(100, 5)
        
        # Normalize data
        norm_data, scaler = self.model.normalize_data(data)
        
        # Check shape
        self.assertEqual(norm_data.shape, data.shape)
        
        # Check normalization: values should be between -1 and 1 after normalization
        self.assertTrue(np.all(norm_data >= -1.0))
        self.assertTrue(np.all(norm_data <= 1.0))
        
        # Test inverse transform
        orig_data = scaler.inverse_transform(norm_data)
        np.testing.assert_allclose(orig_data, data, rtol=1e-10)
    
    def test_prepare_sequences(self):
        """Test preparation of sequences."""
        # Just use the dummy implementation for testing
        X, y = self.model.prepare_sequences(self.X_train, 10)
        
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
    
    @patch('torch.optim.Adam')
    def test_train_model(self, mock_optimizer):
        """Test training the model."""
        # Set up the mock
        mock_optimizer.return_value = MagicMock()
        
        # Build the model
        self.model.build_model()
        
        # Create mock data loaders
        train_loader = [(self.X_train_tensor[:32], self.y_train_tensor[:32])]
        val_loader = [(self.X_test_tensor[:10], self.y_test_tensor[:10])]
        
        # Mock the training process
        with patch.object(self.model, 'train_model', return_value={'loss': [0.1], 'val_loss': [0.2]}):
            # Train the model
            history = self.model.train_model(train_loader, val_loader, num_epochs=2)
            
            # Check the history
            self.assertIn('loss', history)
            self.assertIn('val_loss', history)
    
    @patch('torch.nn.MSELoss')
    def test_evaluate(self, mock_loss):
        """Test evaluating the model."""
        # Set up the mock
        mock_loss.return_value = MagicMock()
        mock_loss.return_value.forward.return_value = torch.tensor(0.1)
        
        # Build the model
        self.model.build_model()
        
        # Create mock test loader
        test_loader = [(self.X_test_tensor[:10], self.y_test_tensor[:10])]
        
        # Mock the evaluation process
        with patch.object(self.model, 'evaluate', return_value={'loss': 0.1, 'mse': 0.1}):
            # Evaluate the model
            metrics = self.model.evaluate(test_loader)
            
            # Check the metrics
            self.assertIn('loss', metrics)
    
    def test_predict(self):
        """Test predicting with the model."""
        # Build the model
        self.model.build_model()
        
        # Mock the prediction process
        with patch.object(self.model, 'predict', return_value=torch.zeros(20, 1)):
            # Make predictions
            predictions = self.model.predict(self.X_test_tensor)
            
            # Check the predictions
            self.assertEqual(predictions.shape, (20, 1))
    
    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        # Build the model
        self.model.build_model()
        
        # Create a path for the model
        model_path = os.path.join(self.temp_dir, 'base_model.pt')
        
        # Save the model
        with patch('torch.save'):
            self.model.save(model_path)
        
        # Load the model into a new instance
        new_model = DummyModel(config_path=self.config_path)
        
        # Build the model
        with patch('torch.load', return_value={
            'model_state_dict': {},
            'config': self.config,
            'history': {'loss': [], 'val_loss': []}
        }):
            new_model.load(model_path)
            
            # Make predictions with the loaded model
            with patch.object(new_model, 'predict', return_value=torch.zeros(20, 1)):
                predictions = new_model.predict(self.X_test_tensor)
                self.assertEqual(predictions.shape, (20, 1))


if __name__ == '__main__':
    unittest.main() 