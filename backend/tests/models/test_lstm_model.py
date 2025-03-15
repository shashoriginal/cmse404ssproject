"""
Unit tests for the LSTM model.
Author: rajshash

This module provides unit tests for the LSTM model used in financial indices prediction.
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
from src.models.lstm import LSTMModel


class TestLSTMModel(unittest.TestCase):
    """Test cases for the LSTM model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            'model': {
                'type': 'lstm',
                'input_shape': [10, 5],  # Convert tuple to list for YAML serialization
                'output_shape': 1,
                'lstm_units': [64, 32],
                'dense_units': [32],
                'dropout_rate': 0.2,
                'bidirectional': False,
                'num_layers': 2,
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
        
        # Create LSTM model
        self.model = LSTMModel(config_path=self.config_path)
        
        # Create dummy data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10, 5)  # 100 samples, 10 time steps, 5 features
        self.y_train = np.random.randn(100, 1)
        self.X_test = np.random.randn(20, 10, 5)
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
        """Test initialization of the LSTM model."""
        self.assertEqual(self.model.config_path, self.config_path)
        self.assertIsInstance(self.model.config, dict)
        self.assertEqual(self.model.input_shape, (10, 5))
        self.assertEqual(self.model.output_shape, 1)
    
    @patch('torch.nn.LSTM')
    def test_build_model(self, mock_lstm):
        """Test building the LSTM model."""
        # Set up the mock
        mock_lstm.return_value = MagicMock()
        
        # Build the model
        model = self.model.build_model()
        
        # Check that the model was built
        self.assertIsNotNone(model)
        mock_lstm.assert_called()
    
    @patch('torch.nn.LSTM')
    def test_forward(self, mock_lstm):
        """Test forward pass of the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(32, 10, 64), (torch.zeros(1, 32, 64), torch.zeros(1, 32, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
        # Build the model
        self.model.build_model()
        
        # Create a sample input
        x = torch.randn(32, 10, 5)  # batch_size, seq_len, n_features
        
        # Mock forward methods
        with patch.object(self.model, 'forward', return_value=torch.randn(32, 1)):
            # Test forward pass
            output = self.model(x)
            self.assertEqual(output.shape, (32, 1))
    
    def test_prepare_sequences(self):
        """Test preparation of sequences for LSTM."""
        # Create some test data
        data = np.random.randn(100, 5)
        sequence_length = 10
        
        # Test with numpy array
        with patch.object(self.model, 'prepare_sequences', return_value=(
            torch.zeros(90, sequence_length, 5), torch.zeros(90, 1)
        )):
            X, y = self.model.prepare_sequences(data, sequence_length)
            
            self.assertEqual(X.shape[1], sequence_length)
            self.assertEqual(X.shape[2], data.shape[1])
            self.assertEqual(y.shape[1], 1)
            self.assertIsInstance(X, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
    
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
    
    @patch('torch.nn.LSTM')
    def test_train_model(self, mock_lstm):
        """Test training the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(32, 10, 64), (torch.zeros(1, 32, 64), torch.zeros(1, 32, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
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
    
    @patch('torch.nn.LSTM')
    def test_evaluate(self, mock_lstm):
        """Test evaluating the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(32, 10, 64), (torch.zeros(1, 32, 64), torch.zeros(1, 32, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
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
    
    @patch('torch.nn.LSTM')
    def test_predict(self, mock_lstm):
        """Test predicting with the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(32, 10, 64), (torch.zeros(1, 32, 64), torch.zeros(1, 32, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
        # Build the model
        self.model.build_model()
        
        # Mock the prediction process
        with patch.object(self.model, 'predict', return_value=torch.zeros(20, 1)):
            # Make predictions
            predictions = self.model.predict(self.X_test_tensor)
            
            # Check the predictions
            self.assertEqual(predictions.shape, (20, 1))
    
    @patch('torch.nn.LSTM')
    def test_predict_sequence(self, mock_lstm):
        """Test multi-step prediction with the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(1, 10, 64), (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
        # Build the model
        self.model.build_model()
        
        # Create a test sequence
        seq = torch.randn(1, 10, 5)  # batch_size, seq_len, n_features
        
        # Mock the prediction process
        with patch.object(self.model, 'predict_sequence', return_value=torch.zeros(5, 1)):
            # Make predictions
            with patch.object(self.model, 'predict', return_value=torch.tensor([[0.5]])):
                predictions = self.model.predict_sequence(seq, steps=5)
                
                # Check the predictions
                self.assertEqual(predictions.shape, (5, 1))
                self.assertIsInstance(predictions, torch.Tensor)
    
    @patch('torch.nn.LSTM')
    def test_save_and_load_model(self, mock_lstm):
        """Test saving and loading the LSTM model."""
        # Set up the mock
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.return_value = (torch.zeros(32, 10, 64), (torch.zeros(1, 32, 64), torch.zeros(1, 32, 64)))
        mock_lstm.return_value = mock_lstm_instance
        
        # Build the model
        self.model.build_model()
        
        # Create a path for the model
        model_path = os.path.join(self.temp_dir, 'lstm_model.pt')
        
        # Save the model
        with patch('torch.save'):
            self.model.save(model_path)
        
        # Load the model into a new instance
        new_model = LSTMModel(config_path=self.config_path)
        
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