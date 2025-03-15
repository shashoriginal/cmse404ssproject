"""
Unit tests for training utilities.
Author: rajshash

This module provides unit tests for the training utilities used in financial indices prediction.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.training import (
    load_and_prepare_data, train_lstm_model, evaluate_and_visualize, run_experiment
)
from src.models.lstm import LSTMModel


class TestTrainingUtilities(unittest.TestCase):
    """Test cases for training utilities."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        self.sequence_length = 10
        
        # Create a test DataFrame
        dates = pd.date_range(start='2020-01-01', periods=self.n_samples)
        data = np.random.randn(self.n_samples, self.n_features)
        self.test_df = pd.DataFrame(data, index=dates, columns=[f'Feature{i+1}' for i in range(self.n_features)])
        
        # Save test data to CSV
        self.data_path = os.path.join(self.test_dir, 'test_data.csv')
        self.test_df.to_csv(self.data_path)
        
        # Create a temporary models directory
        self.models_dir = os.path.join(self.test_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create a temporary results directory
        self.results_dir = os.path.join(self.test_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create a temporary plots directory
        self.plots_dir = os.path.join(self.test_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Save a test model config
        self.model_config = {
            'type': 'lstm',
            'input_shape': [self.sequence_length, self.n_features],
            'output_shape': 1,
            'lstm_units': [64, 32],
            'dense_units': [32],
            'dropout_rate': 0.2,
            'bidirectional': False,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 2
        }
        
        # Create a test LSTM model with patched training
        self.model = LSTMModel()
        self.model.build_model = MagicMock()
        self.model.train_model = MagicMock(return_value={'loss': [0.1, 0.05], 'val_loss': [0.2, 0.1]})
        self.model.evaluate = MagicMock(return_value={'loss': 0.1, 'mse': 0.1})
        self.model.predict = MagicMock(return_value=torch.randn(20, 1))
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_load_and_prepare_data(self):
        """Test loading and preparing data."""
        # Test with valid data path
        train_loader, test_loader, scaler = load_and_prepare_data(
            self.data_path,
            sequence_length=self.sequence_length,
            train_test_split=0.8,
            standardize=True,
            batch_size=32
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        self.assertIsNotNone(scaler)
        
        # Check batch size
        for batch_x, batch_y in train_loader:
            self.assertLessEqual(batch_x.shape[0], 32)  # Batch size
            self.assertEqual(batch_x.shape[1], self.sequence_length)  # Sequence length
            break
        
        # Test with invalid data path
        with self.assertRaises(FileNotFoundError):
            load_and_prepare_data(
                'nonexistent_file.csv',
                sequence_length=self.sequence_length,
                train_test_split=0.8
            )
    
    def test_train_lstm_model(self):
        """Test training an LSTM model."""
        # Load and prepare data
        train_loader, test_loader, _ = load_and_prepare_data(
            self.data_path,
            sequence_length=self.sequence_length,
            train_test_split=0.8,
            standardize=True,
            batch_size=32
        )
        
        # Train the model
        with patch('src.models.lstm.LSTMModel') as mock_lstm:
            # Set up the mock to return our pre-configured model
            mock_lstm.return_value = self.model
            
            trained_model, history = train_lstm_model(
                train_loader,
                val_loader=test_loader,
                input_shape=[self.sequence_length, self.n_features],
                lstm_units=64,
                dense_units=32,
                dropout_rate=0.2,
                bidirectional=False,
                learning_rate=0.001,
                epochs=2
            )
            
            # Check the result
            self.assertIsNotNone(trained_model)
            self.assertIsNotNone(history)
            self.assertIn('loss', history)
            self.assertIn('val_loss', history)
            
            # Check that the model was called with the right parameters
            mock_lstm.assert_called()
            self.model.train_model.assert_called()
    
    def test_evaluate_and_visualize(self):
        """Test evaluating and visualizing model results."""
        # Load and prepare data
        train_loader, test_loader, _ = load_and_prepare_data(
            self.data_path,
            sequence_length=self.sequence_length,
            train_test_split=0.8,
            standardize=True,
            batch_size=32
        )
        
        # Patch the plotting function to avoid displaying plots during tests
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                # Evaluate and visualize
                metrics, predictions, true_values = evaluate_and_visualize(
                    self.model,
                    test_loader,
                    self.plots_dir
                )
                
                # Check the results
                self.assertIsNotNone(metrics)
                self.assertIsNotNone(predictions)
                self.assertIsNotNone(true_values)
                self.assertIn('loss', metrics)
                self.assertIn('mse', metrics)
    
    def test_run_experiment(self):
        """Test running a complete experiment."""
        # Patch the necessary functions
        with patch('src.models.training.load_and_prepare_data') as mock_load:
            with patch('src.models.training.train_lstm_model') as mock_train:
                with patch('src.models.training.evaluate_and_visualize') as mock_eval:
                    # Set up the mocks
                    mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
                    mock_train.return_value = (self.model, {'loss': [0.1], 'val_loss': [0.2]})
                    mock_eval.return_value = (
                        {'loss': 0.1, 'mse': 0.1},
                        torch.randn(20, 1),
                        torch.randn(20, 1)
                    )
                    
                    # Run the experiment
                    results = run_experiment(
                        data_path=self.data_path,
                        sequence_length=self.sequence_length,
                        lstm_units=64,
                        dense_units=32,
                        dropout_rate=0.2,
                        bidirectional=False,
                        learning_rate=0.001,
                        epochs=2,
                        output_dir=self.results_dir,
                        models_dir=self.models_dir
                    )
                    
                    # Check the calls
                    mock_load.assert_called()
                    mock_train.assert_called()
                    mock_eval.assert_called()
                    
                    # Check the results
                    self.assertIsNotNone(results)
                    self.assertIn('test_metrics', results)
                    self.assertIn('predictions', results)
                    self.assertIn('true_values', results)
                    self.assertIn('model_config', results)


if __name__ == '__main__':
    unittest.main() 